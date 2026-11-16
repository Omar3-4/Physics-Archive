import os
import re
import subprocess
import smtplib
from datetime import datetime
from email.message import EmailMessage
import pandas as pd
import requests

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
SUBSCRIBERS = [e.strip() for e in os.environ["SUBSCRIBERS_LIST"].split(",") if e.strip()]
CSV_FILE = "History of Physics - Papers.csv"

def get_next_paper(df):
    if "Status" not in df.columns:
        df["Status"] = ""
    if "Sent_Date" not in df.columns:
        df["Sent_Date"] = ""
    
    for idx, row in df.iterrows():
        status = str(row["Status"]).strip().upper()
        if status != "SENT":
            return idx, row
    return None, None

def call_gemini(prompt):
    models = ["gemini-3.8-flash", "gemini-4-hydrogen", "gemini-3-flash-preview", "gemini-3.5-flash-lite"]
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 32768}
        }
        res = requests.post(url, json=payload, timeout=180)
        if res.status_code == 200:
            data = res.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "\n".join(p["text"] for p in parts if "text" in p and not p.get("thought", False)).strip()
            if len(text) > 100:
                print(f"✅ Generated via {model} ({len(text)} chars)")
                return text
        else:
            print(f"⚠️ {model} failed with HTTP {res.status_code}")
    raise RuntimeError("All Gemini models failed.")

def main():
    df = pd.read_csv(CSV_FILE, dtype=str).fillna("")
    idx, paper = get_next_paper(df)
    if paper is None:
        print("All papers have been sent!")
        return

    paper_id = paper["ID"]
    date = paper["Exact Date"]
    scientist = paper["Scientist"]
    title = paper["Original Title & English Translation"]
    venue = paper["Publication Venue / Academy Archive"]
    puzzle = paper["The Puzzle Piece (Specific Contribution)"]
    bridge = paper["Causal Bridge (Before -> After)"]
    link = paper["Primary Archival Link / Reference"]

    prompt = f"""
أنت أستاذ فيزياء نظرية وتاريخ علوم متخصص في التحليل الرياضي الصارم للنصوص التأسيسية.
مهمتك اليوم هي شرح هذه الورقة/العمل العلمي المحدد:
- الرمز الأرشيفي: {paper_id}
- التاريخ: {date}
- العالم: {scientist}
- العنوان: {title}
- جهة النشر/الأرشيف: {venue}
- الرابط المرجعي المساعد: {link}
- السياق في سلسلة البازل: {puzzle} | {bridge}

**بروتوكول التقيّد بالمصدر الأصلي (إلزامي):**
اعتمد في شرحك على البنية الفعلية للعمل الأصلي ومنهج مؤلفه الحقيقي (أبواب الكتاب، القضايا Propositions، الفرضيات، والمعادلات/الهندسة المستخدمة فيه) لا على التلخيصات الحديثة السطحية.

**اشرح لي هذه الورقة بهذا الأسلوب الأكاديمي الصارم:**
أريد عرضًا دقيقًا ومفصلًا يبدأ بـ:
1. **التأصيل التاريخي المنهجي للفكرة:** السياق العلمي الدقيق والمأزق المعرفي أو الرياضي الذي سبق هذه الورقة مباشرة، وكيف ارتبطت بالأعمال التي قبلها.
2. **التعريفات الرسمية (Formal Definitions):** عرض دقيق للمصطلحات والكميات الفيزيائية كما عرّفها المؤلف في الورقة، مع توضيح تطور دلالتها.
3. **تحليل الفرضيات والبديهيات (Axioms & Postulates):** تفكيك المسلمات الفيزيائية والهندسية/الرياضية التي بنى عليها المؤلف استدلاله داخل النص.
4. **البرهنة الكاملة خطوة بخطوة (Rigorous Step-by-Step Derivation & Proof):** تتبع البرهان الرياضي أو الهندسي الكامل للمفاهيم والنظريات الواردة في الورقة، مع الحفاظ على التسلسل المنطقي والصرامة الرياضية في كل خطوة (تجنب التبسيط أو الاستعانة بالأمثلة السطحية، وركّز على البناء المفهومي العميق، حتى لو استدعى ذلك الدخول في التفاصيل التقنية المعقدة).
5. **التمثيل والتطبيق العملي (Physical Application & Mathematical Modeling):** تطبيق البرهان على مسألة فيزيائية محددة أو نموذج رياضي يوضح القوة التفسيرية للورقة.

**قواعد صارمة للإخراج والتوثيق:**
- ابدأ الشرح مباشرة بالعنوان الأول دون أي مقدمات ترحيبية.
- اجعل إجابتك باللغة العربية الفصحى الرصينة (والإنجليزية فقط للمصطلحات العلمية، وعناوين الأوراق، والاقتباسات الأصلية).
- استخدم صيغة Markdown + LaTeX القياسية لجميع المعادلات ($...$ للمعادلات السطرية و $$...$$ للمعادلات المستقلة).
- عند ذكر أي تجربة، نظرية، أو معادلة أنتجها عالم معين، يجب دائماً الإشارة إلى:
  * اسم الورقة العلمية أو الكتاب الأصلي بدقة.
  * رقم الصفحة أو الفصل/القضية (Book / Chapter / Proposition / Theorem) التي ورد فيها ذلك داخل المصدر الأصلي.
  * الاسم العلمي المتعارف عليه للتجربة (إن وجدت).
  * اقتباس حرفي قصير من النص الأصلي بلغته الأصلية مع ترجمته العربية.
"""

    lecture_body = call_gemini(prompt)

    os.makedirs("lectures_md", exist_ok=True)
    os.makedirs("lectures_pdf", exist_ok=True)

    clean_author = re.sub(r'[^a-zA-Z0-9]', '_', scientist.split('(')[0].strip())
    clean_date = re.sub(r'[^0-9\-]', '', str(date))
    base_name = f"{paper_id}_{clean_date}_{clean_author}"

    md_path = f"lectures_md/{base_name}.md"
    pdf_path = f"lectures_pdf/{base_name}.pdf"

    full_markdown = f"""---
id: "{paper_id}"
date: "{date}"
scientist: "{scientist}"
title: "{title}"
venue: "{venue}"
link: "{link}"
---

# \\textenglish{{{title}}}

* **المؤلف:** \\textenglish{{{scientist}}} | **التاريخ:** {date}
* **جهة النشر:** \\textenglish{{{venue}}}
* **قطعة البازل:** \\textenglish{{{puzzle}}}
* **الجسر السببي:** \\textenglish{{{bridge}}}
* **الرابط الأرشيفي:** <{link}>

---

{lecture_body}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    pandoc_cmd = [
        "pandoc", md_path,
        "-o", pdf_path,
        "--pdf-engine=xelatex",
        "-H", "header.tex",
        "-V", "dir=rtl"
    ]
    subprocess.run(pandoc_cmd, check=True)
    print(f"✅ Compiled Vector PDF: {pdf_path}")

    msg = EmailMessage()
    msg["Subject"] = f"[{paper_id} | {date}] {scientist.split('(')[0].strip()}: محاضرة تأسيسية (XeLaTeX PDF)"
    msg["From"] = f"Physics Archive <{GMAIL_USER}>"
    msg["To"] = SUBSCRIBERS[0]
    if len(SUBSCRIBERS) > 1:
        msg["Bcc"] = ", ".join(SUBSCRIBERS[1:])

    msg.set_content(
        f"محاضرة اليوم من أرشيف الفيزياء التأسيسية:\n\n"
        f"العمل: {title}\n"
        f"المؤلف: {scientist} ({date})\n"
        f"قطعة البازل: {puzzle}\n"
        f"الجسر السببي: {bridge}\n"
        f"الرابط الأرشيفي: {link}\n\n"
        f"مرفق طيه نسخة PDF مطبوعة بمحرك XeLaTeX ونسخة Markdown لـ Obsidian."
    )

    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=f"{base_name}.pdf")
    with open(md_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="text", subtype="markdown", filename=f"{base_name}.md")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)
    print("✅ Email sent successfully!")

    df.at[idx, "Status"] = "SENT"
    df.at[idx, "Sent_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
