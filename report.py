from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.pagesizes import A4

from datetime import datetime

import os



# ============================================
# REPORT GENERATION FUNCTION
# ============================================

def generate_report(
        patient,
        age,
        prediction,
        confidence
):


    # Create reports folder

    report_folder = "static/reports"


    os.makedirs(

        report_folder,

        exist_ok=True

    )


    # PDF file path

    filename = (

        patient.replace(" ","_")

        +

        "_DR_Report.pdf"

    )


    filepath = os.path.join(

        report_folder,

        filename

    )



    # Create PDF

    pdf = SimpleDocTemplate(

        filepath,

        pagesize=A4

    )



    styles = getSampleStyleSheet()


    content = []



    # ====================================
    # TITLE
    # ====================================

    content.append(

        Paragraph(

            "Diabetic Retinopathy Detection Report",

            styles["Title"]

        )

    )


    content.append(

        Spacer(1,20)

    )



    # ====================================
    # PATIENT DETAILS
    # ====================================

    details = f"""

    <b>Patient Name:</b> {patient}<br/>

    <b>Age:</b> {age}<br/>

    <b>Date:</b> {datetime.now().strftime("%d-%m-%Y %H:%M")}<br/>

    """



    content.append(

        Paragraph(

            details,

            styles["Normal"]

        )

    )


    content.append(

        Spacer(1,20)

    )



    # ====================================
    # AI RESULT
    # ====================================

    result = f"""

    <b>AI Prediction:</b> {prediction}<br/>

    <b>Confidence Score:</b> {confidence}%<br/>

    """



    content.append(

        Paragraph(

            result,

            styles["Normal"]

        )

    )



    content.append(

        Spacer(1,20)

    )



    # ====================================
    # OBSERVATION
    # ====================================

    observations = {

        "No DR":
        "No diabetic retinopathy signs detected. Regular eye check-up is recommended.",


        "Mild":
        "Mild diabetic retinopathy detected. Periodic monitoring is advised.",


        "Moderate":
        "Moderate diabetic retinopathy detected. Consultation with an eye specialist is recommended.",


        "Severe":
        "Severe diabetic retinopathy detected. Immediate medical consultation is advised.",


        "Proliferative DR":
        "High-risk diabetic retinopathy detected. Urgent ophthalmologist consultation is recommended."

    }


    observation_text = observations.get(

        prediction,

        "Please consult a medical professional."

    )



    content.append(

        Paragraph(

            "<b>AI Observation:</b><br/>"

            + observation_text,

            styles["Normal"]

        )

    )


    content.append(

        Spacer(1,30)

    )



    # ====================================
    # DISCLAIMER
    # ====================================

    disclaimer = """

    <b>Disclaimer:</b><br/>

    This report is generated using an AI-based screening system.

    It is intended for assistance only and should not replace

    professional medical diagnosis.

    """



    content.append(

        Paragraph(

            disclaimer,

            styles["Normal"]

        )

    )



    # Build PDF

    pdf.build(

        content

    )



    return filepath