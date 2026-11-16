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

def extract_and_run_diagram_code(raw_text, img_path):
    """يستخرج كود Matplotlib من رد Gemini وينفذه لإنتاج رسمة هندسية عالية الدقة"""
    match = re.search(r'```python\s*([\s\S]*?)```', raw_text)
    if not match:
        return raw_text, False
    
    code = match.group(1)
    # إزالة بلوك الكود من نص المحاضرة حتى لا يظهر ككود برمجي داخل الـ PDF
    clean_text = raw_text.replace(match.group(0), "").strip()
    
    # إجبار الكود على الحفظ في المسار المحدد دون فتح نافذة رسومية
    header_code = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
    code = re.sub(r'plt\.show\(\)', '', code)
    code = re.sub(r'plt\.savefig\([^\)]+\)', f"plt.savefig('{img_path}', dpi=300, bbox_inches='tight', facecolor='white')", code)
    if "plt.savefig" not in code:
        code += f"\nplt.savefig('{img_path}', dpi=300, bbox_inches='tight', facecolor='white')\n"
    
    with open("temp_plot.py", "w", encoding="utf-8") as f:
        f.write(header_code + code)
    
    res = subprocess.run(["python", "temp_plot.py"], capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(img_path):
        print(f"✅ Generated geometric diagram: {img_path}")
        return clean_text, True
    else:
        print(f"⚠️ Diagram generation skipped due to plot error: {res.stderr[-300:]}")
        return clean_text, False

def sanitize_markdown_for_xelatex(text):
    text = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'([^\n])\n\$\$', r'\1\n\n$$', text) text = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', text)

    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$)', text)
    math_cmds = r'(\\(?:sin\vert{}cos\vert{}tan\vert{}cot\vert{}sec\vert{}csc\vert{}arcsin\vert{}arccos\vert{}arctan\vert{}ln\vert{}log\vert{}exp\vert{}lim\vert{}sum\vert{}int\vert{}prod\vert{}sqrt\vert{}frac\vert{}partial\vert{}nabla\vert{}infty\vert{}alpha\vert{}beta\vert{}gamma\vert{}Gamma\vert{}delta\vert{}Delta\vert{}epsilon\vert{}varepsilon\vert{}zeta\vert{}eta\vert{}theta\vert{}Theta\vert{}iota\vert{}kappa\vert{}lambda\vert{}Lambda\vert{}mu\vert{}nu\vert{}xi\vert{}Xi\vert{}pi\vert{}Pi\vert{}rho\vert{}sigma\vert{}Sigma\vert{}tau\vert{}upsilon\vert{}phi\vert{}Phi\vert{}varphi\vert{}chi\vert{}psi\vert{}Psi\vert{}omega\vert{}Omega\vert{}cdot\vert{}times\vert{}pm\vert{}mp\vert{}leq\vert{}geq\vert{}neq\vert{}approx\vert{}equiv\vert{}propto\vert{}rightarrow\vert{}Rightarrow\vert{}leftarrow\vert{}Leftarrow\vert{}vec\vert{}hat\vert{}bar\vert{}dot\vert{}ddot)\b(?:\{[^}]*\}\vert{}\^\{[^}]*\}\vert{}_\{[^}]*\}\vert{}\^[a-zA-Z0-9]\vert{}_[a-zA-Z0-9])*)'  for i in range(0, len(parts), 2): parts[i] = re.sub(math_cmds, r'$\1$', parts[i]) parts[i] = re.sub(r'(?<!\\)_', r'\\_', parts[i])

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
1. **منع الاقتباسات الأجنبية نهائياً:** لا تكتب أي اقتباس حرفي باللاتينية أو الفرنسية أو الألمانية. اكتب الشرح بالكامل باللغة العربية الفصحى الرصينة (مع الإبقاء فقط على المصطلحات العلمية الإنجليزية بين قوسين).
2. **ترجمة عنوان الورقة للغتين:** في أول سطرين من إجابتك، اكتب:
   - **عنوان العمل بالإنجليزية:** \\textenglish{{(الترجمة الإنجليزية الدقيقة للعنوان)}}
   - **عنوان العمل بالعربية:** (الترجمة العربية الأكاديمية الدقيقة للعنوان)
3. **شرح كل رمز رياضي وهندسي (Symbol Legend):** أي رمز تستخدمه في أي معادلة أو برهان يجب أن توضح معناه الفيزيائي/الهندسي بدقة فور ظهوره في قائمة واضحة.
4. **التخيّل البصري والهندسي الدقيق:** اشرح التكوين الهندسي بدقة متناهية (المراكز، المحاور، الزوايا، المسارات) ليرى القارئ البنية الرياضية بوضوح تام.
5. **توليد رسمة هندسية/فيزيائية عبر Python (Matplotlib):** في نهاية إجابتك تماماً، اكتب بلوك كود بايثون واحد محصور بين ` ```python ` و ` ``` ` يستخدم مكتبة `matplotlib` و `numpy` لرسم الشكل الهندسي أو المخطط الفيزيائي الجوهري لهذه الورقة بدقة عالية (مع تسمية النقاط والمتجهات والزوايا باللغة الإنجليزية مثل $O, C, D, P, R, \\theta$، وإضافة عنوان واضح للشكل، وحفظه باستخدام `plt.savefig('diagram.png')`).

**اشرح لي هذه الورقة وفق هذه المحاور الستة بالترتيب:**
1. **بطاقة العنوان والهوية**
2. **التأصيل التاريخي المنهجي للفكرة**
3. **التعريفات الرسمية ودليل الرموز (Formal Definitions & Symbol Guide)**
4. **تحليل الفرضيات والبديهيات (Axioms & Postulates)**
5. **البرهنة الكاملة خطوة بخطوة مع الوصف التخيلي للشكل**
6. **التمثيل والتطبيق العملي**

**قواعد التنسيق:**
- ابدأ الشرح مباشرة بالمحور الأول دون أي مقدمات ترحيبية.
- لا تكتب أي كلمة عربية إطلاقاً داخل علامات الرياضيات `$ ... $` أو `$$ ... $$` (اكتب الوحدات مثل "يوماً" أو "راديان" خارج علامات الدولار).
- استخدم صيغة Markdown + LaTeX القياسية لجميع المعادلات (`$ ... $` و `$$ ... $$`).
"""

    raw_response = call_gemini(prompt)

    os.makedirs("lectures_md", exist_ok=True)
    os.makedirs("lectures_pdf", exist_ok=True)
    os.makedirs("lectures_img", exist_ok=True)

    clean_author = re.sub(r'[^a-zA-Z0-9]', '_', scientist.split('(')[0].strip())
    clean_date = re.sub(r'[^0-9\-]', '', str(date))
    base_name = f"{paper_id}_{clean_date}_{clean_author}"

    md_path = f"lectures_md/{base_name}.md"
    pdf_path = f"lectures_pdf/{base_name}.pdf"
    img_path = f"lectures_img/{base_name}.png"

    raw_lecture_body, has_diagram = extract_and_run_diagram_code(raw_response, img_path)
    clean_lecture_body = sanitize_markdown_for_xelatex(raw_lecture_body)

    safe_title = title.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_scientist = scientist.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_venue = venue.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_puzzle = puzzle.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_bridge = bridge.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    diagram_latex_block = ""
    if has_diagram:
        diagram_latex_block = f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.88\\textwidth]{{{img_path}}}
\\caption{{\\textenglish{{Geometric & Physical Diagram — {paper_id}}}}}
\\end{{figure}}
"""

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
{diagram_latex_block}
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

    diagram_md_block = f"\n![Geometric Diagram](../{img_path})\n" if has_diagram else ""
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
{diagram_md_block}
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
        f"مرفق طيه نسخة PDF مطبوعة بمحرك XeLaTeX (مدعومة بالرسم الهندسي) ونسخة Markdown لـ Obsidian."
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
