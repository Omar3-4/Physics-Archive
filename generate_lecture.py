import os
import re
import sys
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
        res = requests.post(url, json=payload, timeout=240)
        if res.status_code == 200:
            data = res.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "\n".join(p["text"] for p in parts if "text" in p and not p.get("thought", False)).strip()
            if len(text) > 100:
                print(f"✅ Generated via {model} ({len(text)} chars)")
                return text
        else:
            print(f"WARNING {model} failed with HTTP {res.status_code}")
    raise RuntimeError("All Gemini models failed.")


def extract_and_run_diagram_code(raw_text, img_path):
    match = re.search(r'```python\s*([\s\S]*?)```', raw_text)
    if not match:
        return raw_text, False

    code = match.group(1)
    clean_text = raw_text.replace(match.group(0), "").strip()

    header_code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
    )
    code = re.sub(r'plt\.show\(\)', '', code)
    code = re.sub(
        r'plt\.savefig\([^\)]+\)',
        f"plt.savefig('{img_path}', dpi=300, bbox_inches='tight', facecolor='white')",
        code
    )
    if "plt.savefig" not in code:
        code += f"\nplt.savefig('{img_path}', dpi=300, bbox_inches='tight', facecolor='white')\n"

    with open("temp_plot.py", "w", encoding="utf-8") as f:
        f.write(header_code + code)

    try:
        res = subprocess.run([sys.executable, "temp_plot.py"], capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and os.path.exists(img_path):
            print(f"Generated geometric diagram: {img_path}")
            return clean_text, True
        else:
            print(f"WARNING Diagram skipped due to plot error: {res.stderr[-300:]}")
            return clean_text, False
    except subprocess.TimeoutExpired:
        print("WARNING Diagram skipped: execution timed out after 60s.")
        return clean_text, False


def generate_interactive_html(paper_id, date, scientist, title, lecture_body):
    html_prompt = f"""
You are an expert computational physicist and scientific visualization engineer.
Based on the historical physics lecture provided below for:
- Paper: {title} ({date}) by {scientist}

Build a **single-file, self-contained, interactive HTML5 Canvas demonstration** that visually animates and demonstrates the exact mathematical/geometric derivation step-by-step using ONLY period-accurate tools from {date}.

**STRICT DESIGN & CONTENT RULES:**
1. **NO PROSE OR EXPLANATORY PARAGRAPHS:** Do NOT write descriptive text, paragraphs, or wordy explanations. Communicate 100% visually through:
   - An interactive dark-mode HTML5 `<canvas>` (`#090d16` background) drawing the exact geometric/physical construction with labeled points, vectors, angles, and highlighted active elements.
   - Interactive sliders to control key system parameters (e.g., angles, eccentricity, radii, time) plus Play/Pause and Step Next/Prev buttons.
   - Live KaTeX parameter readouts above the canvas.
   - A right-hand panel containing ONLY pure symbolic step-by-step derivation cards rendered with KaTeX (`https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css` & `.js`) and Tailwind CSS (`https://cdn.tailwindcss.com`). Clicking a derivation step or advancing steps must highlight the corresponding geometric elements/angles on the canvas.
2. **Historical Fidelity:** Use ONLY the exact symbols, geometric constructions, and mathematical methods from {date} present in the lecture below.
3. **Output Format:** Return ONLY valid, complete HTML starting with `<!DOCTYPE html>` and ending with `</html>`. Do not wrap in markdown fences or add any text outside the HTML.

**LECTURE CONTENT TO VISUALIZE:**
{lecture_body[:18000]}
"""
    try:
        raw_html = call_gemini(html_prompt)
        match = re.search(r'(<!DOCTYPE html[\s\S]*?</html>)', raw_html, re.IGNORECASE)
        if match:
            return match.group(1)
        if "<html" in raw_html.lower():
            return raw_html
    except Exception as e:
        print(f"WARNING HTML generation failed: {e}")
    return None


def sanitize_markdown_for_xelatex(text):
    text = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'([^\n])\n\$\$', r'\1\n\n$$', text)
    text = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', text)

    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$)', text)

    cmd_list = [
        "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
        "ln", "log", "exp", "lim", "sum", "int", "prod", "sqrt", "frac", "partial",
        "nabla", "infty", "alpha", "beta", "gamma", "Gamma", "delta", "Delta",
        "epsilon", "varepsilon", "zeta", "eta", "theta", "Theta", "iota", "kappa",
        "lambda", "Lambda", "mu", "nu", "xi", "Xi", "pi", "Pi", "rho", "sigma",
        "Sigma", "tau", "upsilon", "phi", "Phi", "varphi", "chi", "psi", "Psi",
        "omega", "Omega", "cdot", "times", "pm", "mp", "leq", "geq", "neq",
        "approx", "equiv", "propto", "rightarrow", "Rightarrow", "leftarrow",
        "Leftarrow", "vec", "hat", "bar", "dot", "ddot"
    ]
    cmd_names = "|".join(cmd_list)

    brace_arg = r'\{(?:[^{}]|\{[^{}]*\})*\}'
    math_cmds = re.compile(
        r'(\\(?:' + cmd_names + r')\b'
        r'(?:' + brace_arg + r'|\^' + brace_arg + r'|_' + brace_arg + r'|\^[a-zA-Z0-9]|_[a-zA-Z0-9])*)'
    )

    for i in range(0, len(parts), 2):
        parts[i] = math_cmds.sub(r'$\1$', parts[i])
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
أنت مؤرخ علوم وأستاذ فيزياء نظرية. مهمتك شرح هذا العمل العلمي لطالب فيزياء يريد أن يفهم كيف فكر المؤلف وبرهن بأدوات عصره فقط.

بيانات الورقة:
- الرمز الأرشيفي: {paper_id}
- التاريخ: {date}
- العالم: {scientist}
- العنوان: {title}
- جهة النشر: {venue}
- الإسهام المحدد: {puzzle}
- الرابط السببي بين الأوراق: {bridge}
- الرابط الأرشيفي: {link}

**قواعد الأسلوب والمحتوى (إلزامية):**

1. **أدوات عصر المؤلف فقط:**
   - يُمنع استخدام أي أداة رياضية أو مفهوم فيزيائي لم يكن موجوداً قبل تاريخ {date}.
   - استخدم المنهج الرياضي الفعلي الذي وظّفه المؤلف في نصه.

2. **اللغة العربية:**
   - اكتب بعربية علمية واضحة ومباشرة. تجنب الكلمات القديمة أو النادرة التي تحتاج إلى قاموس.
   - لا تستخدم أي رموز تعبيرية (emoji) في أي موضع.
   - لا تكتب مقدمات ترحيبية أو عبارات مبالغة وصفية. ابدأ المحتوى مباشرة.
   - المصطلحات العلمية: اكتبها بالإنجليزية بين قوسين داخل الأمر `\\textenglish{{(...)}}` مثل: `\\textenglish{{(Heliocentric)}}`.

3. **قواعد LaTeX:**
   - لا تكتب أي نص عربي داخل `$ ... $` أو `$$ ... $$`. الوحدات مثل "يوم" تُكتب خارج علامات الرياضيات.
   - أي نص إنجليزي أو رقم بين قوسين في سياق النص العربي: ضعه داخل `\\textenglish{{(...)}}`. مثال: `\\textenglish{{(1473--1543)}}`. استخدم -- للشرطة المعترضة، لا — .
   - كل رمز رياضي يجب أن يظهر تعريفه قبل استخدامه.
   - الجداول: عمودان فقط `| الرمز | الدلالة |` وسطر واحد لكل رمز.
   - استخدم `$$` على سطر مستقل للمعادلات المستقلة، و`$...$` للمعادلات الداخلية.
   - استخدم علامة  على سطر مستقل للمعادلات المستقلة.

4. **الرسم الهندسي (Matplotlib):**
   - في نهاية الإجابة بعد المحور العاشر مباشرة، اكتب بلوك Python واحد بين ` ```python ` و ` ``` `.
   - استخدم `matplotlib` و `numpy` لرسم التكوين الهندسي الجوهري.
   - سمّ النقاط والزوايا بالإنجليزية. أضف عنواناً للشكل. احفظه بـ `plt.savefig('diagram.png')`.

**اشرح الورقة وفق هذه المحاور العشرة بالترتيب:**

1. **بطاقة الهوية**
   - الرمز الأرشيفي، المؤلف، التاريخ، جهة النشر، موضع الورقة في سلسلة البازل.
   - أول سطرين: **عنوان العمل بالإنجليزية:** \\textenglish{{(...)}} ثم **عنوان العمل بالعربية:** الترجمة المباشرة.

2. **السياق التاريخي**
   - ما المشكلة التي كانت عالقة قبل هذا العمل؟ ما الأدوات والأفكار المتاحة وقت كتابته؟

3. **دليل الرموز**
   - جدول بكل رمز هندسي أو فيزيائي سيُستخدم في البراهين.

4. **الفرضيات والبديهيات**
   - ما الافتراضات الأساسية التي انطلق منها المؤلف؟

5. **البرهان الكامل خطوة بخطوة**
   - باستخدام أدوات المؤلف فقط. صف التكوين الهندسي بدقة قبل كل برهان.

6. **التطبيق العددي**
   - تطبيق رقمي على حالة محددة بالأرقام التي استخدمها المؤلف أو التي كانت متاحة في عصره.

7. **صوت المؤلف**
   - اقتبس بالترجمة العربية المباشرة جملة أو فقرة واحدة من النص الأصلي للمؤلف، مع تحديد الكتاب والفصل أو الصفحة. إذا لم تتذكر نصاً محدداً، اذكر الموضع الذي توجد فيه الحجة الجوهرية بالضبط.

8. **العدسة الحديثة**
   - في 3 إلى 5 أسطر فقط: كيف تُعبّر عن النتيجة الجوهرية ذاتها اليوم بأدوات حساب المتجهات أو التفاضل والتكامل؟ الهدف إظهار الفارق وليس استبدال البرهان التاريخي.

9. **الثغرة التي لم يرها المؤلف**
   - افتراض واحد أو خطأ واحد محدد في هذه الورقة، اكتشفه العلم لاحقاً. حدد من كشفه وفي أي عمل.

10. **السؤال المعلَّق**
    - سؤال واحد مفتوح لا إجابة له في هذه الورقة، يظهر منه التناقض أو الثغرة التي ستحلها الورقة التالية ({bridge}). اكتبه من منظور من يعيش في عصر المؤلف ولا يعرف الحل بعد.
"""

    raw_response = call_gemini(prompt)

    os.makedirs("lectures_md", exist_ok=True)
    os.makedirs("lectures_pdf", exist_ok=True)
    os.makedirs("lectures_img", exist_ok=True)
    os.makedirs("lectures_html", exist_ok=True)

    clean_author = re.sub(r'[^a-zA-Z0-9]', '_', scientist.split('(')[0].strip())
    clean_date = re.sub(r'[^0-9\-]', '', str(date))
    base_name = f"{paper_id}_{clean_date}_{clean_author}"

    md_path = f"lectures_md/{base_name}.md"
    pdf_path = f"lectures_pdf/{base_name}.pdf"
    img_path = f"lectures_img/{base_name}.png"
    html_path = f"lectures_html/{base_name}.html"

    raw_lecture_body, has_diagram = extract_and_run_diagram_code(raw_response, img_path)
    clean_lecture_body = sanitize_markdown_for_xelatex(raw_lecture_body)

    # توليد ملف الـ HTML التفاعلي بناءً على المحاضرة
    interactive_html = generate_interactive_html(paper_id, date, scientist, title, raw_lecture_body)
    has_html = False
    if interactive_html:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(interactive_html)
        has_html = True
        print(f"Generated Interactive HTML: {html_path}")

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
\\caption{{\\textenglish{{Geometric and Physical Diagram — {paper_id}}}}}
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
        print("WARNING XeLaTeX syntax issue, retrying without raw_tex...")
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

    print(f"Compiled Vector PDF: {pdf_path}")

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
    msg["Subject"] = f"[{paper_id} | {date}] {scientist.split('(')[0].strip()}: محاضرة تأسيسية + محاكاة تفاعلية"
    msg["From"] = f"Physics Archive <{GMAIL_USER}>"
    msg["To"] = SUBSCRIBERS[0]
    if len(SUBSCRIBERS) > 1:
        msg["Bcc"] = ", ".join(SUBSCRIBERS[1:])

    msg.set_content(
        f"[{paper_id}] {scientist.split('(')[0].strip()} -- {date}\n\n"
        f"{title}\n\n"
        f"الإسهام: {puzzle}\n\n"
        f"الرابط السببي: {bridge}\n\n"
        f"المصدر الأصلي: {link}\n\n"
        f"المرفقات: PDF (XeLaTeX) + Markdown (Obsidian) + HTML تفاعلي."
    )

    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=f"{base_name}.pdf")
    with open(md_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="text", subtype="markdown", filename=f"{base_name}.md")
    if has_html and os.path.exists(html_path):
        with open(html_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="text", subtype="html", filename=f"{base_name}.html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)
    print("Email sent.")

    df.at[idx, "Status"] = "SENT"
    df.at[idx, "Sent_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
