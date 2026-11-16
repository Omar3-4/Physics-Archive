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
    models = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite"]
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

def sanitize_markdown_for_xelatex(text):
    # 1. تحويل صيغ اللاتكس البديلة \[...\] و \(...\) إلى $$...$$ و $...$ القياسية لـ Pandoc
    text = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    # 2. التأكد من وجود سطر فارغ قبل وبعد المعادلات المستقلة $$...$$
    text = re.sub(r'([^\n])\n\$\$', r'\1\n\n$$', text)
    text = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', text)

    # 3. حماية الأوامر الرياضية الشائعة خارج علامات $...$
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$)', text)
    math_cmds = r'(\\(?:sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|ln|log|exp|lim|sum|int|prod|sqrt|frac|partial|nabla|infty|alpha|beta|gamma|Gamma|delta|Delta|epsilon|varepsilon|zeta|eta|theta|Theta|iota|kappa|lambda|Lambda|mu|nu|xi|Xi|pi|Pi|rho|sigma|Sigma|tau|upsilon|phi|Phi|varphi|chi|psi|Psi|omega|Omega|cdot|times|pm|mp|leq|geq|neq|approx|equiv|propto|rightarrow|Rightarrow|leftarrow|Leftarrow|vec|hat|bar|dot|ddot)\b(?:\{[^}]*\}|\^\{[^}]*\}|_\{[^}]*\}|\^[a-zA-Z0-9]|_[a-zA-Z0-9])*)'
    
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(math_cmds, r'$\1$', parts[i])
        parts[i] = re.sub(r'(?<!\\)_', r'\\_', parts[i])

    return "".join(parts)

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
مهمتك اليوم هي شرح هذه الورقة/العمل العلمي المحدد لطالب فيزياء متخصص يريد فهماً هندسياً ورياضياً عميقاً بدون أي غموض:
- الرمز الأرشيفي: {paper_id}
- التاريخ: {date}
- العالم: {scientist}
- العنوان المسجل: {title}
- جهة النشر/الأرشيف: {venue}
- السياق في سلسلة البازل: {puzzle} | {bridge}

**قواعد صارمة جداً للأسلوب والمحتوى (إلزامية):**
1. **منع الاقتباسات الأجنبية نهائياً:** لا تكتب أي اقتباس حرفي باللاتينية أو الفرنسية أو الألمانية أو غيرها. اكتب الشرح بالكامل باللغة العربية الفصحى الرصينة (مع الإبقاء فقط على المصطلحات العلمية الإنجليزية بين قوسين).
2. **ترجمة عنوان الورقة للغتين:** في أول سطرين من إجابتك، اكتب:
   - **عنوان العمل بالإنجليزية:** (الترجمة الإنجليزية الدقيقة للعنوان).
   - **عنوان العمل بالعربية:** (الترجمة العربية الأكاديمية الدقيقة للعنوان).
3. **شرح كل رمز رياضي وهندسي (Symbol Legend):** أي رمز تستخدمه في أي معادلة أو برهان (مثل $r, \\theta, \\omega, C, D, \\Gamma_1, \\dots$) يجب أن توضح معناه الفيزيائي/الهندسي بدقة فور ظهوره في قائمة واضحة، ولا تترك أي رمز مجهول الدلالة.
4. **التخيّل البصري والهندسي الدقيق (Spatial & Geometric Visualization):** بما أن القارئ يقرأ نصاً بدون الرسومات المرفقة في المخطوطة الأصلية، يجب عليك في كل خطوة هندسية أو فيزيائية أن تشرح له **كيف يتخيل الشكل في ذهنه بدقة** (أين يقع المركز؟ ما هي الدائرة الثابتة وأيها المتحركة؟ كيف تتغير الزوايا والأطوال مع الزمن؟ ما المسار الناتج؟). تجنب الأمثلة السطحية أو التشبيهات العامية، وركز على الوصف الهندسي والفضائي الدقيق الذي يجعل القارئ يرى البنية الرياضية بوضوح تام.

**اشرح لي هذه الورقة وفق هذه المحاور الستة بالترتيب:**
1. **بطاقة العنوان والهوية:** (الترجمة الإنجليزية والعربية لعنوان الورقة + المرجع الدقيق للأبواب/القضايا Propositions داخل العمل الأصلي).
2. **التأصيل التاريخي المنهجي للفكرة:** السياق العلمي الدقيق والمأزق المعرفي أو الرياضي الذي سبق هذه الورقة مباشرة، وكيف ارتبطت بالأعمال التي قبلها.
3. **التعريفات الرسمية ودليل الرموز (Formal Definitions & Symbol Guide):** عرض دقيق للمصطلحات والكميات الفيزيائية كما عرّفها المؤلف، مع جدول أو قائمة شاملة تشرح كل رمز رياضي وهندسي سيُستخدم في الورقة.
4. **تحليل الفرضيات والبديهيات (Axioms & Postulates):** تفكيك المسلمات الفيزيائية والهندسية/الرياضية التي بنى عليها المؤلف استدلاله داخل النص.
5. **البرهنة الكاملة خطوة بخطوة مع الوصف التخيلي للشكل (Rigorous Step-by-Step Proof & Geometric Visualization):** تتبع البرهان الرياضي أو الهندسي الكامل للمفاهيم والنظريات الواردة في الورقة، مع دمج **شرح التخيّل البصري والهندسي** لكل نقطة وخط ومسار في كل خطوة من خطوات البرهان، والحفاظ على الصرامة الرياضية الكاملة.
6. **التمثيل والتطبيق العملي (Physical Application & Mathematical Modeling):** تطبيق البرهان على مسألة فيزيائية محددة أو نموذج رياضي يوضح القوة التفسيرية للورقة.

**قواعد التنسيق:**
- ابدأ الشرح مباشرة بالمحور الأول دون أي مقدمات ترحيبية.
- استخدم صيغة Markdown + LaTeX القياسية لجميع المعادلات: ضع أي رمز رياضي أو دالة مثل $\\sin\\theta$ دائماً بين علامتي دولار مفردتين `$ ... $`، وضع المعادلات المستقلة بين علامتي دولار مزدوجتين `$$ ... $$` مع ترك سطر فارغ قبلهما وبعدهما. لا تستخدم `\\(` أو `\\[`.
- لا تكتب أي كلمة عربية إطلاقاً داخل علامات الرياضيات `$ ... $` أو `$$ ... $$` (اكتب الوحدات مثل "يوماً" أو "راديان" خارج علامات الدولار)، وضع العنوان الإنجليزي بين قوسين معكوفين `\textenglish{...}` حتى لا ينعكس اتجاه كلماته.
"""

    raw_lecture_body = call_gemini(prompt)
    clean_lecture_body = sanitize_markdown_for_xelatex(raw_lecture_body)

    os.makedirs("lectures_md", exist_ok=True)
    os.makedirs("lectures_pdf", exist_ok=True)

    clean_author = re.sub(r'[^a-zA-Z0-9]', '_', scientist.split('(')[0].strip())
    clean_date = re.sub(r'[^0-9\-]', '', str(date))
    base_name = f"{paper_id}_{clean_date}_{clean_author}"

    md_path = f"lectures_md/{base_name}.md"
    pdf_path = f"lectures_pdf/{base_name}.pdf"

    safe_title = title.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_scientist = scientist.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_venue = venue.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_puzzle = puzzle.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_bridge = bridge.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    full_markdown = f"""---
id: "{paper_id}"
date: "{date}"
scientist: "{scientist}"
title: "{title}"
---

# \\textenglish{{{safe_title}}}

* **المؤلف:** \\textenglish{{{safe_scientist}}} | **التاريخ:** {date}
* **جهة النشر:** \\textenglish{{{safe_venue}}}
* **قطعة البازل:** \\textenglish{{{safe_puzzle}}}
* **الجسر السببي:** \\textenglish{{{safe_bridge}}}
* **الرابط الأرشيفي:** <{link}>

---

{clean_lecture_body}
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
    result = subprocess.run(pandoc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️ XeLaTeX encountered a syntax issue, retrying with raw_tex disabled as fallback...")
        print(result.stderr[-500:])
        fallback_cmd = [
            "pandoc", md_path,
            "-f", "markdown-raw_tex",
            "-o", pdf_path,
            "--pdf-engine=xelatex",
            "-H", "header.tex",
            "-V", "dir=rtl"
        ]
        subprocess.run(fallback_cmd, check=True)

    print(f"✅ Compiled Vector PDF: {pdf_path}")

    obsidian_md = f"""---
id: "{paper_id}"
date: "{date}"
scientist: "{scientist}"
title: "{title}"
venue: "{venue}"
link: "{link}"
---

# {title}

* **المؤلف:** {scientist} | **التاريخ:** {date}
* **جهة النشر:** {venue}
* **قطعة البازل:** {puzzle}
* **الجسر السببي:** {bridge}
* **الرابط الأرشيفي:** <{link}>

---

{raw_lecture_body}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(obsidian_md)

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
