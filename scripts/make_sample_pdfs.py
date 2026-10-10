"""Generate 3 sample Thai PDFs for the eval harness and demo.

Each fact used by eval.py lives in its own paragraph so it maps cleanly
to a retrieval chunk.
"""
from __future__ import annotations

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

FONT = os.path.expanduser("~/.fonts/Sarabun-Regular.ttf")
pdfmetrics.registerFont(TTFont("Thai", FONT))

style = ParagraphStyle("th", fontName="Thai", fontSize=14, leading=22, spaceAfter=10)
title_style = ParagraphStyle("th-title", fontName="Thai", fontSize=20, leading=28, spaceAfter=14)

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_pdfs")


def write_pdf(filename: str, title: str, paragraphs: list[str]) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, filename)
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    story = [Paragraph(title, title_style), Spacer(1, 0.3 * cm)]
    for p in paragraphs:
        story.append(Paragraph(p, style))
    doc.build(story)
    return path


DOCS = {
    "leave-policy.pdf": (
        "นโยบายการลางาน พ.ศ. 2569",
        [
            "พนักงานประจำที่มีอายุงานครบ 1 ปี มีสิทธิลาพักร้อนได้ 12 วันทำการต่อปี "
            "เมื่ออายุงานครบ 5 ปี สิทธิจะเพิ่มเป็น 15 วันทำการต่อปี โดยวันลาที่เหลือสามารถสะสมไปใช้ในปีถัดไปได้ไม่เกิน 6 วัน",

            "พนักงานมีสิทธิลาป่วยได้ 30 วันทำการต่อปี โดยได้รับค่าจ้างเต็มจำนวน "
            "หากลาป่วยติดต่อกันเกิน 3 วัน ต้องแนบใบรับรองแพทย์ประกอบการลา",

            "พนักงานมีสิทธิลากิจได้ 6 วันทำการต่อปี โดยได้รับค่าจ้าง "
            "การลากิจเพื่อธุระจำเป็นเร่งด่วนสามารถแจ้งหัวหน้างานทางโทรศัพท์ก่อน แล้วทำเอกสารย้อนหลังภายใน 2 วันทำการ",

            "พนักงานหญิงมีสิทธิลาคลอดบุตรได้ 98 วันต่อครรภ์ โดยได้รับค่าจ้างจากบริษัท 45 วัน "
            "ส่วนที่เหลือสามารถเบิกจากกองทุนประกันสังคมตามกฎหมาย",

            "การลาพักร้อนต้องยื่นใบลาล่วงหน้าอย่างน้อย 3 วันทำการ และต้องได้รับการอนุมัติจากหัวหน้างานโดยตรงก่อนจึงจะมีผล",
        ],
    ),
    "product-manual.pdf": (
        "คู่มือการใช้งาน SiamRAG",
        [
            "SiamRAG คือเว็บแอปพลิเคชันถาม-ตอบเอกสารภาษาไทย ผู้ใช้อัปโหลดไฟล์ PDF แล้วถามคำถามเป็นภาษาไทย "
            "ระบบจะค้นหาข้อความที่เกี่ยวข้องและตอบพร้อมอ้างอิงแหล่งที่มาทุกครั้ง",

            "SiamRAG มี 3 แพ็กเกจ ได้แก่ Starter ราคา 990 บาทต่อเดือน รองรับเอกสารสูงสุด 50 ฉบับ "
            "แพ็กเกจ Pro ราคา 2,900 บาทต่อเดือน รองรับเอกสารสูงสุด 500 ฉบับ "
            "และแพ็กเกจ Enterprise สำหรับองค์กรขนาดใหญ่ กรุณาติดต่อฝ่ายขายเพื่อขอใบเสนอราคา",

            "ข้อจำกัดของไฟล์: รองรับเฉพาะไฟล์ PDF ขนาดไม่เกิน 50 เมกะไบต์ต่อไฟล์ "
            "เอกสารที่มีข้อความแบบรูปภาพ (สแกน) จะยังอ่านไม่ได้ในเวอร์ชันปัจจุบัน",

            "ผู้ใช้ใหม่สามารถทดลองใช้ฟรี 14 วัน โดยไม่ต้องกรอกข้อมูลบัตรเครดิต "
            "เมื่อหมดช่วงทดลอง ระบบจะปรับเป็นแพ็กเกจ Starter อัตโนมัติหากไม่มีการยกเลิก",

            "ฝ่ายสนับสนุนลูกค้าให้บริการวันจันทร์ถึงวันศุกร์ เวลา 9:00 ถึง 18:00 น. "
            "ช่องทางติดต่อ: อีเมล support@siamrag.example และไลน์ @siamrag",
        ],
    ),
    "bangkok-guide.pdf": (
        "คู่มือท่องเที่ยวกรุงเทพมหานคร",
        [
            "รถไฟฟ้า BTS สายสีเขียวมีค่าโดยสารเริ่มต้นที่ 17 บาท และสูงสุดที่ 62 บาทต่อเที่ยว "
            "ผู้ถือบัตรแรบบิทแบบรายเดือนจะได้รับส่วนลดเพิ่มเติมตามจำนวนเที่ยวที่ใช้",

            "วัดพระศรีรัตนศาสดาราม (วัดพระแก้ว) เปิดให้เข้าชมทุกวัน เวลา 8:30 ถึง 15:30 น. "
            "ค่าเข้าชมสำหรับนักท่องเที่ยวชาวต่างชาติ 500 บาท ส่วนคนไทยเข้าชมฟรีโดยแสดงบัตรประชาชน",

            "ตลาดนัดจตุจักรเปิดเฉพาะวันเสาร์และวันอาทิตย์ เวลา 9:00 ถึง 18:00 น. "
            "มีร้านค้ามากกว่า 15,000 ร้าน แบ่งเป็น 27 โซน ควรเผื่อเวลาอย่างน้อยครึ่งวัน",

            "การเดินทางจากสนามบินสุวรรณภูมิเข้าเมืองที่สะดวกที่สุดคือรถไฟฟ้า Airport Rail Link "
            "ค่าโดยสาร 45 บาท ใช้เวลาเดินทางประมาณ 30 นาทีถึงสถานีพญาไท",
        ],
    ),
}


def main() -> None:
    for filename, (title, paras) in DOCS.items():
        path = write_pdf(filename, title, paras)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
