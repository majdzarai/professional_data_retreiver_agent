#!/usr/bin/env python3
"""
PDF Generator Agent

This agent processes writer output and generates a comprehensive PDF report
with all chapters in the correct order according to the report structure.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from reportlab.pdfgen import canvas
from io import BytesIO
import re
from html import unescape

logger = logging.getLogger(__name__)

class PDFGenerator:
    """
    Agent that processes writer output and generates a comprehensive PDF report.
    """
    
    def __init__(self, output_dir: str = "output/pdf"):
        """
        Initialize the PDF Generator.
        
        Args:
            output_dir (str): Directory to save generated PDF
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDF Generator initialized with output directory: {self.output_dir}")
    
    def generate_pdf_report(self, writer_dir: str, structure_path: str) -> str:
        """
        Generate a comprehensive PDF report from writer output.
        
        Args:
            writer_dir (str): Directory containing writer output
            structure_path (str): Path to report structure JSON file
            
        Returns:
            str: Path to generated PDF file
        """
        # Store writer output directory for use in other methods
        self.writer_output_dir = writer_dir
        writer_path = Path(writer_dir)
        
        # Load report structure to get correct chapter order
        with open(structure_path, 'r', encoding='utf-8') as f:
            structure = json.load(f)
        
        # Load report summary for metadata
        summary_file = writer_path / "report_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                report_summary = json.load(f)
        else:
            report_summary = {"report_generation_summary": {"total_chapters": 0, "total_sections": 0}}
        
        # Generate HTML content
        html_content = self._generate_html_content(writer_path, structure, report_summary)
        
        # Generate PDF
        pdf_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = self.output_dir / pdf_filename
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory confirmed: {self.output_dir.absolute()}")
        logger.info(f"PDF will be saved to: {pdf_path.absolute()}")
        
        # Generate PDF using ReportLab
        self._generate_reportlab_pdf(pdf_path, structure, report_summary)
        
        # Final verification
        if pdf_path.exists():
            logger.info(f"PDF report generated successfully: {pdf_path}")
        else:
            logger.error(f"PDF was not created at expected location: {pdf_path}")
            
        return str(pdf_path)
    
    def _generate_html_content(self, writer_path: Path, structure: Dict[str, Any], 
                             report_summary: Dict[str, Any]) -> str:
        """
        Generate HTML content for the PDF report.
        
        Args:
            writer_path: Path to writer output directory
            structure: Report structure dictionary
            report_summary: Report generation summary
            
        Returns:
            str: Complete HTML content
        """
        chapters_dir = writer_path / "chapters"
        
        # Start HTML document
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{structure.get('report_title', 'Report')}</title>",
            "</head>",
            "<body>"
        ]
        
        # Add title page
        html_parts.extend(self._generate_title_page(structure, report_summary))
        
        # Add table of contents
        html_parts.extend(self._generate_table_of_contents(structure, chapters_dir))
        
        # Add chapters in correct order
        for chapter in structure.get("chapters", []):
            chapter_id = chapter.get("id")
            chapter_title = chapter.get("title")
            
            # Find corresponding markdown file
            chapter_filename = self._create_safe_filename(chapter_title) + ".md"
            chapter_file = chapters_dir / chapter_filename
            
            if chapter_file.exists():
                logger.info(f"Adding chapter: {chapter_title}")
                
                # Read markdown content
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(markdown_content, extensions=['tables', 'toc'])
                
                # Add page break before each chapter (except first)
                html_parts.append('<div class="page-break"></div>')
                html_parts.append(f'<div class="chapter" id="chapter-{chapter_id}">')
                html_parts.append(html_content)
                html_parts.append('</div>')
            else:
                logger.warning(f"Chapter file not found: {chapter_file}")
        
        # Close HTML document
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _generate_title_page(self, structure: Dict[str, Any], 
                           report_summary: Dict[str, Any]) -> List[str]:
        """
        Generate title page HTML.
        
        Args:
            structure: Report structure dictionary
            report_summary: Report generation summary
            
        Returns:
            List[str]: HTML parts for title page
        """
        summary_data = report_summary.get("report_generation_summary", {})
        
        return [
            '<div class="title-page">',
            f'<h1 class="report-title">{structure.get("report_title", "Report")}</h1>',
            '<div class="report-metadata">',
            f'<p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y")}</p>',
            f'<p><strong>Total Chapters:</strong> {summary_data.get("total_chapters", "N/A")}</p>',
            f'<p><strong>Total Sections:</strong> {summary_data.get("total_sections", "N/A")}</p>',
            f'<p><strong>Total Words:</strong> {summary_data.get("total_word_count", "N/A"):,}</p>',
            '</div>',
            '</div>'
        ]
    
    def _generate_table_of_contents(self, structure: Dict[str, Any], 
                                  chapters_dir: Path) -> List[str]:
        """
        Generate table of contents HTML.
        
        Args:
            structure: Report structure dictionary
            chapters_dir: Path to chapters directory
            
        Returns:
            List[str]: HTML parts for table of contents
        """
        toc_parts = [
            '<div class="page-break"></div>',
            '<div class="table-of-contents">',
            '<h2>Table of Contents</h2>',
            '<ul class="toc-list">'
        ]
        
        for i, chapter in enumerate(structure.get("chapters", []), 1):
            chapter_id = chapter.get("id")
            chapter_title = chapter.get("title")
            
            # Check if chapter file exists
            chapter_filename = self._create_safe_filename(chapter_title) + ".md"
            chapter_file = chapters_dir / chapter_filename
            
            if chapter_file.exists():
                toc_parts.append(
                    f'<li class="toc-item">'
                    f'<span class="chapter-number">{i}.</span> '
                    f'<a href="#chapter-{chapter_id}">{chapter_title}</a>'
                    f'</li>'
                )
        
        toc_parts.extend(['</ul>', '</div>'])
        return toc_parts
    
    def _generate_reportlab_pdf(self, pdf_path: Path, 
                               structure: Dict[str, Any], report_summary: Dict[str, Any]):
        """
        Generate PDF using ReportLab.
        
        Args:
            html_content: HTML content to convert
            pdf_path: Path to save PDF
            structure: Report structure
            report_summary: Report summary data
        """
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = self._create_reportlab_styles()
        
        # Build story (content)
        story = []
        
        # Add title page
        story.extend(self._create_title_page_reportlab(structure, report_summary, styles))
        story.append(PageBreak())
        
        # Add table of contents
        story.extend(self._create_toc_reportlab(structure, styles))
        story.append(PageBreak())
        
        # Add chapters from writer output
        writer_dir = Path(self.writer_output_dir)
        chapters_dir = writer_dir / "chapters"
        
        if chapters_dir.exists():
            # Get chapter order from structure
            chapter_order = [chapter.get("title") for chapter in structure.get("chapters", [])]
            
            for chapter in structure.get("chapters", []):
                  chapter_title = chapter.get("title")
                  chapter_filename = self._create_safe_filename(chapter_title) + ".md"
                  chapter_file = chapters_dir / chapter_filename
                  
                  if chapter_file.exists():
                      with open(chapter_file, 'r', encoding='utf-8') as f:
                          markdown_content = f.read()
                      
                      # Add chapter title
                      story.append(Paragraph(chapter_title, styles['ChapterTitle']))
                      story.append(Spacer(1, 0.3*inch))
                      
                      # Process sections according to structure
                      sections = chapter.get("sections", [])
                      if sections:
                          # Parse content by sections
                          section_content = self._parse_content_by_sections(markdown_content, sections)
                          for section in sections:
                              section_title = section.get("title")
                              section_text = section_content.get(section_title, "")
                              
                              if section_text:
                                  # Add section title
                                  story.append(Paragraph(section_title, styles['SectionTitle']))
                                  story.append(Spacer(1, 0.2*inch))
                                  
                                  # Convert section content to flowables
                                  html_content = self._convert_markdown_to_html(section_text)
                                  section_flowables = self._html_to_flowables(html_content, styles)
                                  story.extend(section_flowables)
                                  story.append(Spacer(1, 0.2*inch))
                      else:
                          # No sections defined, process entire chapter content
                          html_content = self._convert_markdown_to_html(markdown_content)
                          chapter_flowables = self._html_to_flowables(html_content, styles)
                          story.extend(chapter_flowables)
                      

                      
                      story.append(PageBreak())
                  else:
                      logger.warning(f"Chapter file not found: {chapter_file}")
        
        # Build PDF with error handling
        try:
            logger.info(f"Building PDF with {len(story)} elements...")
            doc.build(story)
            logger.info(f"PDF successfully built and saved to: {pdf_path}")
            
            # Verify file was created
            if pdf_path.exists():
                file_size = pdf_path.stat().st_size
                logger.info(f"PDF file created successfully. Size: {file_size} bytes")
            else:
                logger.error(f"PDF file was not created at: {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error building PDF: {str(e)}")
            raise
    
    def _create_reportlab_styles(self):
        """
        Create ReportLab styles for the PDF.
        
        Returns:
            StyleSheet: ReportLab styles
        """
        styles = getSampleStyleSheet()
        
        # Define custom styles with unique names
        custom_styles = {
            'ReportTitle': ParagraphStyle(
                name='ReportTitle',
                parent=styles['Heading1'],
                fontSize=28,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#2c3e50')
            ),
            'ReportSubtitle': ParagraphStyle(
                name='ReportSubtitle',
                parent=styles['Normal'],
                fontSize=16,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#7f8c8d')
            ),
            'ChapterTitle': ParagraphStyle(
                name='ChapterTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceBefore=20,
                spaceAfter=15,
                textColor=colors.HexColor('#2c3e50'),
                keepWithNext=1
            ),
            'SectionTitle': ParagraphStyle(
                name='SectionTitle',
                parent=styles['Heading2'],
                fontSize=18,
                spaceBefore=15,
                spaceAfter=10,
                textColor=colors.HexColor('#34495e'),
                keepWithNext=1
            ),
            'SubsectionTitle': ParagraphStyle(
                name='SubsectionTitle',
                parent=styles['Heading3'],
                fontSize=14,
                spaceBefore=12,
                spaceAfter=8,
                textColor=colors.HexColor('#34495e'),
                keepWithNext=1
            ),
            'BodyText': ParagraphStyle(
                name='BodyText',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                textColor=colors.HexColor('#333333')
            ),
            'TOCHeading': ParagraphStyle(
                name='TOCHeading',
                parent=styles['Heading1'],
                fontSize=24,
                spaceBefore=0,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#2c3e50'),
                fontName='Helvetica-Bold'
            ),
            'TOCChapter': ParagraphStyle(
                name='TOCChapter',
                parent=styles['Normal'],
                fontSize=14,
                spaceAfter=6,
                spaceBefore=8,
                leftIndent=20,
                textColor=colors.HexColor('#2c3e50'),
                fontName='Helvetica-Bold'
            ),
            'TOCSection': ParagraphStyle(
                name='TOCSection',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=4,
                leftIndent=40,
                textColor=colors.HexColor('#34495e')
            ),
            'TOCEntry': ParagraphStyle(
                name='TOCEntry',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=5,
                leftIndent=0
            )
        }
        
        # Add custom styles to stylesheet
        for style_name, style_obj in custom_styles.items():
            if style_name not in styles:
                styles.add(style_obj)
        
        return styles
    
    def _create_title_page_reportlab(self, structure: Dict[str, Any], 
                                   report_summary: Dict[str, Any], styles) -> List:
        """
        Create title page using ReportLab.
        """
        story = []
        summary_data = report_summary.get("report_generation_summary", {})
        
        # Add some space at top
        story.append(Spacer(1, 2*inch))
        
        # Title
        story.append(Paragraph(structure.get("report_title", "Report"), styles['ReportTitle']))
        
        # Metadata
        metadata_text = f"""
        <b>Generated:</b> {datetime.now().strftime("%B %d, %Y")}<br/>
        <b>Total Chapters:</b> {summary_data.get("total_chapters", "N/A")}<br/>
        <b>Total Sections:</b> {summary_data.get("total_sections", "N/A")}<br/>
        <b>Total Words:</b> {summary_data.get("total_word_count", "N/A"):,}
        """
        story.append(Paragraph(metadata_text, styles['ReportSubtitle']))
        
        return story
    
    def _create_toc_reportlab(self, structure: Dict[str, Any], styles) -> List:
        """
        Create a comprehensive table of contents using ReportLab with chapters and sections.
        """
        story = []
        
        # TOC heading with enhanced styling
        story.append(Paragraph("Table of Contents", styles['TOCHeading']))
        story.append(Spacer(1, 20))
        
        # TOC entries with chapters and sections
        for i, chapter in enumerate(structure.get("chapters", []), 1):
            chapter_title = chapter.get("title", "")
            
            # Chapter entry with bold formatting and proper spacing
            chapter_entry = f"<b>{i}. {chapter_title}</b>"
            story.append(Paragraph(chapter_entry, styles['TOCChapter']))
            
            # Add sections if they exist
            sections = chapter.get("sections", [])
            if sections:
                for j, section in enumerate(sections, 1):
                    section_title = section.get("title", "")
                    section_entry = f"&nbsp;&nbsp;&nbsp;&nbsp;{i}.{j} {section_title}"
                    story.append(Paragraph(section_entry, styles['TOCSection']))
            
            # Add spacing after each chapter
            story.append(Spacer(1, 8))
        
        # Add page break to ensure TOC takes full page
        story.append(PageBreak())
        
        return story
    
    def _convert_markdown_to_html(self, markdown_content: str) -> str:
        """
        Convert markdown content to HTML.
        
        Args:
            markdown_content (str): Markdown content
            
        Returns:
            str: HTML content
        """
        # Convert markdown to HTML
        md = markdown.Markdown(
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.toc',
                'markdown.extensions.nl2br'
            ]
        )
        
        return md.convert(markdown_content)
    
    def _parse_content_by_sections(self, markdown_content: str, sections: List) -> dict:
        """
        Parse markdown content and extract text for each section.
        
        Args:
            markdown_content (str): Full markdown content of the chapter
            sections (List): List of section definitions from structure JSON
            
        Returns:
            dict: Dictionary mapping section titles to their content
        """
        section_content = {}
        lines = markdown_content.split('\n')
        current_section = None
        current_content = []
        
        # Create a mapping of section titles for matching
        section_titles = {section.get("title"): section for section in sections}
        
        for line in lines:
            # Check if this line is a section header (## format)
            if line.startswith('## '):
                # Save previous section content if any
                if current_section and current_content:
                    section_content[current_section] = '\n'.join(current_content).strip()
                
                # Extract section title from header
                header_title = line[3:].strip()
                
                # Check if this header matches any of our defined sections
                if header_title in section_titles:
                    current_section = header_title
                    current_content = []
                else:
                    current_section = None
                    current_content = []
            elif current_section:
                # Add line to current section content
                current_content.append(line)
        
        # Don't forget the last section
        if current_section and current_content:
            section_content[current_section] = '\n'.join(current_content).strip()
        
        return section_content
    
    def _html_to_flowables(self, html_content: str, styles, chapter_title: str = None) -> List:
        """
        Convert HTML content to ReportLab flowables.
        
        Args:
            html_content: HTML content
            styles: ReportLab styles
            chapter_title: Title of the chapter
            
        Returns:
            List: ReportLab flowables
        """
        story = []
        
        # Add chapter title at the beginning
        if chapter_title:
            story.append(Paragraph(chapter_title, styles['ChapterTitle']))
            story.append(Spacer(1, 0.2*inch))
        
        # Simple HTML parsing - split by common tags
        lines = html_content.split('\n')
        current_paragraph = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if para_text:
                        story.append(Paragraph(para_text, styles['BodyText']))
                    current_paragraph = []
                story.append(Spacer(1, 0.1*inch))
                i += 1
                continue
            
            # Handle headings
            if line.startswith('<h1>'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if para_text:
                        story.append(Paragraph(para_text, styles['BodyText']))
                    current_paragraph = []
                text = re.sub(r'<[^>]+>', '', line)
                if text and text != chapter_title:  # Avoid duplicate chapter titles
                    story.append(Paragraph(text, styles['SectionTitle']))
                    story.append(Spacer(1, 0.1*inch))
            elif line.startswith('<h2>'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if para_text:
                        story.append(Paragraph(para_text, styles['BodyText']))
                    current_paragraph = []
                text = re.sub(r'<[^>]+>', '', line)
                if text:
                    story.append(Paragraph(text, styles['SectionTitle']))
                    story.append(Spacer(1, 0.1*inch))
            elif line.startswith('<h3>'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if para_text:
                        story.append(Paragraph(para_text, styles['BodyText']))
                    current_paragraph = []
                text = re.sub(r'<[^>]+>', '', line)
                if text:
                    story.append(Paragraph(text, styles['SubsectionTitle']))
                    story.append(Spacer(1, 0.05*inch))
            elif line.startswith('<h4>'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if para_text:
                        story.append(Paragraph(para_text, styles['BodyText']))
                    current_paragraph = []
                text = re.sub(r'<[^>]+>', '', line)
                if text:
                    story.append(Paragraph(f"<b>{text}</b>", styles['BodyText']))
                    story.append(Spacer(1, 0.05*inch))
            elif line.startswith('<p>'):
                text = re.sub(r'<[^>]+>', '', line)
                text = unescape(text)
                if text:
                    current_paragraph.append(text)
            elif line.startswith('<ul>') or line.startswith('<ol>'):
                pass  # Skip list containers
            elif line.startswith('<li>'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if para_text:
                        story.append(Paragraph(para_text, styles['BodyText']))
                    current_paragraph = []
                text = re.sub(r'<[^>]+>', '', line)
                text = unescape(text)
                if text:
                    story.append(Paragraph(f"• {text}", styles['BodyText']))
            elif not line.startswith('<'):
                # Plain text or markdown-style content
                text = unescape(line)
                if text:
                    # Handle markdown-style headers
                    if text.startswith('# '):
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            if para_text:
                                story.append(Paragraph(para_text, styles['BodyText']))
                            current_paragraph = []
                        header_text = text[2:].strip()
                        if header_text and header_text != chapter_title:
                            story.append(Paragraph(header_text, styles['SectionTitle']))
                            story.append(Spacer(1, 0.1*inch))
                    elif text.startswith('## '):
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            if para_text:
                                story.append(Paragraph(para_text, styles['BodyText']))
                            current_paragraph = []
                        header_text = text[3:].strip()
                        story.append(Paragraph(header_text, styles['SubsectionTitle']))
                        story.append(Spacer(1, 0.05*inch))
                    elif text.startswith('### '):
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            if para_text:
                                story.append(Paragraph(para_text, styles['BodyText']))
                            current_paragraph = []
                        header_text = text[4:].strip()
                        story.append(Paragraph(f"<b>{header_text}</b>", styles['BodyText']))
                        story.append(Spacer(1, 0.05*inch))
                    elif text.startswith('- ') or text.startswith('* '):
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            if para_text:
                                story.append(Paragraph(para_text, styles['BodyText']))
                            current_paragraph = []
                        bullet_text = text[2:].strip()
                        story.append(Paragraph(f"• {bullet_text}", styles['BodyText']))
                    else:
                        current_paragraph.append(text)
                
                i += 1
        
        # Handle remaining paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            if para_text:
                story.append(Paragraph(para_text, styles['BodyText']))
        
        return story
    

     
    def _create_safe_filename(self, title: str) -> str:
        """
        Create a safe filename from a title.
        This method matches the Writer Agent's filename generation logic.
        
        Args:
            title: Original title
            
        Returns:
            str: Safe filename
        """
        # Replace spaces and special characters (matching Writer Agent logic)
        safe_name = title.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        # Remove other problematic characters
        safe_chars = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
        return safe_chars[:50]  # Limit length