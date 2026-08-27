import io
import json
from datetime import datetime
from .schemas import AnalysisResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
import base64
from PIL import Image

def generate_pdf_report(response: AnalysisResponse, query: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=HexColor('#003366'), spaceAfter=14)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=HexColor('#006699'), spaceBefore=12, spaceAfter=6)
    normal_style = styles['Normal']
    code_style = ParagraphStyle('CodeStyle', parent=styles['Code'], fontSize=9, leading=11, textColor=HexColor('#333333'))
    
    elements = []
    
    elements.append(Paragraph("SATQUERY AI — MISSION ANALYSIS REPORT", title_style))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().isoformat() + 'Z'}", normal_style))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("OVERVIEW", heading_style))
    elements.append(Paragraph(f"<b>Query:</b> {query}", normal_style))
    elements.append(Paragraph(f"<b>Task:</b> {response.task}", normal_style))
    elements.append(Paragraph(f"<b>Status:</b> {response.status}", normal_style))
    if response.conflict:
        elements.append(Paragraph("<b>Conflict Detected:</b> Yes", normal_style))
    if response.abstention_reason:
        elements.append(Paragraph(f"<b>Abstention Reason:</b> {response.abstention_reason}", normal_style))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("PRIMARY FINDING", heading_style))
    elements.append(Paragraph(response.answer, normal_style))
    if response.confidence is not None:
        elements.append(Paragraph(f"<b>Confidence:</b> {response.confidence:.2%}", normal_style))
    
    if response.visual_output and response.visual_output.startswith("data:image"):
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("SPATIAL EVIDENCE", heading_style))
        try:
            header, encoded = response.visual_output.split(",", 1)
            image_data = base64.b64decode(encoded)
            img_io = io.BytesIO(image_data)
            img = Image.open(img_io)
            temp_img_path = "temp_visual_output.png"
            img.save(temp_img_path)
            
            rl_img = RLImage(temp_img_path)
            max_width = 6 * inch
            if rl_img.drawWidth > max_width:
                factor = max_width / rl_img.drawWidth
                rl_img.drawWidth = max_width
                rl_img.drawHeight = rl_img.drawHeight * factor
            elements.append(rl_img)
            elements.append(Spacer(1, 10))
        except Exception as e:
            elements.append(Paragraph(f"[Failed to embed visual evidence: {str(e)}]", normal_style))
            
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("SUPPORTING EVIDENCE", heading_style))
    if response.evidence:
        for ev in response.evidence:
            try:
                # Need to handle dictionary or object because Pydantic model vs dict depends on usage
                ev_dict = ev.model_dump() if hasattr(ev, 'model_dump') else ev
                claim = ev_dict.get('claim', 'N/A')
                ev_str = ev_dict.get('evidence', 'N/A')
                modality = ev_dict.get('modality', 'N/A')
                elements.append(Paragraph(f"• <b>{claim}</b> ({modality}): {ev_str}", normal_style))
            except Exception as e:
                elements.append(Paragraph(f"• {str(ev)}", normal_style))
    else:
        elements.append(Paragraph("None provided.", normal_style))
        
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("EXECUTION TRACE", heading_style))
    for step in response.execution_trace:
        elements.append(Paragraph(f"→ {step}", normal_style))
        
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("PROVENANCE", heading_style))
    if response.provenance:
        prov_dict = response.provenance.model_dump() if hasattr(response.provenance, 'model_dump') else response.provenance
        prov_json = json.dumps(prov_dict, indent=2)
        elements.append(Preformatted(prov_json, code_style))
    else:
        elements.append(Paragraph("N/A", normal_style))
        
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("INPUT VALIDATION", heading_style))
    if response.validation:
        for v in response.validation:
            val_dict = v.model_dump() if hasattr(v, 'model_dump') else v
            val_json = json.dumps(val_dict, indent=2)
            elements.append(Preformatted(val_json, code_style))
            elements.append(Spacer(1, 5))
    else:
        elements.append(Paragraph("None.", normal_style))

    doc.build(elements)
    
    import os
    if os.path.exists("temp_visual_output.png"):
        os.remove("temp_visual_output.png")
        
    return buffer.getvalue()
