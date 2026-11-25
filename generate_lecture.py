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


def call_gemini(prompt, max_tokens=32768, temperature=0.25):
    models = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite"]
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }
        try:
            res = requests.post(url, json=payload, timeout=240)
            if res.status_code == 200:
                data = res.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "\n".join(p["text"] for p in parts if "text" in p and not p.get("thought", False)).strip()
                if len(text) > 50:
                    print(f"Generated via {model} ({len(text)} chars)")
                    return text
            else:
                print(f"WARNING {model} failed with HTTP {res.status_code}")
        except Exception as e:
            print(f"WARNING {model} request exception: {e}")
    raise RuntimeError("All Gemini models failed.")


def process_multiple_diagrams(raw_text, base_name):
    """
    يستخرج عدة أكواد رسومات ([[FIG_1]], [[FIG_2]], ...) من رد Gemini،
    ينفذ كل كود لإنتاج صورة مستقلة، ويعيد خريطة بالصور الناجحة.
    """
    fig_blocks = re.findall(r'\[\[(FIG_\d+)\]\]\s*```python\s*([\s\S]*?)```', raw_text)
    clean_text = raw_text
    generated_figs = {}

    for fig_id, code in fig_blocks:
        pattern = re.compile(r'\[\[' + re.escape(fig_id) + r'\]\]\s*```python\s*' + re.escape(code) + r'```')
        clean_text = pattern.sub(f"\n\n[[{fig_id}]]\n\n", clean_text, count=1)

        img_path = f"lectures_img/{base_name}_{fig_id.lower()}.png"
        header_code = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n"
            "plt.rcParams['axes.unicode_minus'] = False\n"
        )
        plot_code = re.sub(r'plt\.show\(\)', '', code)
        plot_code = re.sub(
            r'plt\.savefig\([^\)]+\)',
            f"plt.savefig('{img_path}', dpi=300, bbox_inches='tight', facecolor='white')",
            plot_code
        )
        if "plt.savefig" not in plot_code:
            plot_code += f"\nplt.savefig('{img_path}', dpi=300, bbox_inches='tight', facecolor='white')\n"
        plot_code += "\nplt.close('all')\n"

        with open("temp_plot.py", "w", encoding="utf-8") as f:
            f.write(header_code + plot_code)

        try:
            res = subprocess.run([sys.executable, "temp_plot.py"], capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and os.path.exists(img_path):
                print(f"Generated contextual figure: {img_path}")
                generated_figs[fig_id] = img_path
            else:
                print(f"WARNING {fig_id} skipped due to plot error: {res.stderr[-250:]}")
        except subprocess.TimeoutExpired:
            print(f"WARNING {fig_id} skipped: timeout.")

    clean_text = re.sub(r'```python[\s\S]*?```', '', clean_text).strip()
    return clean_text, generated_figs


def sanitize_markdown_for_xelatex(text):
    # 1. توحيد صيغ الرياضيات
    text = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'([^\n])\n\$\$', r'\1\n\n$$', text)
    text = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', text)

    # 2. إزالة أي أوامر \textenglish سابقة لتجنب التكرار
    text = re.sub(r'\\textenglish\{([^}]*)\}', r'\1', text)

    # 3. اختصار عنوان المحور ليكون "البرهان" فقط
    text = re.sub(r'###\s*البرهان خطوة بخطوة بأدوات عصر المؤلف', r'### البرهان', text)

    # 4. تقسيم النص مع حماية الرياضيات والرسومات والصور
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$|!\[[^\]]*\]\([^)]+\)|\[\[FIG_\d+\]\])', text)

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

    # حماية الأقواس باللغة الإنجليزية في النص العربي لمنع انعكاسها
    ascii_parens = re.compile(r'\(([\x20-\x7E]+)\)')

    for i in range(len(parts)):
        part = parts[i]
        if part.startswith("$") or part.startswith("[[FIG_") or part.startswith("!["):
            continue
        part = math_cmds.sub(r'$\1$', part)
        part = re.sub(r'(?<!\\)_', r'\\_', part)
        part = ascii_parens.sub(r'\\textenglish{(\1)}', part)
        parts[i] = part

    return "".join(parts)


def generate_interactive_html(paper_id, date, scientist, title, lecture_body):
    html_prompt = f"""
You are an expert computational physicist and scientific visualization engineer.
Build a **single-file, self-contained, interactive HTML5 Canvas demonstration** for this historical physics paper:
- Paper: {title} ({date}) by {scientist}

**CRITICAL REQUIREMENTS:**
1. Return a complete, valid, single HTML file starting with `<!DOCTYPE html>` and ending with `</html>`. Do NOT wrap in markdown fences.
2. **KaTeX & Auto-Render:** Include KaTeX CSS & JS and auto-render extension from CDN. Render math equations accurately in both canvas parameter readouts and the step-by-step panel.
3. **Interactive Visual Demonstration:**
   - Left side: HTML5 `<canvas>` (#090d16 dark background) showing animated, interactive geometry with sliders and Play/Pause.
   - Right side: Step-by-step proof cards with concise Arabic explanations for each geometric step.
4. Keep the HTML self-contained, responsive, and working smoothly.

LECTURE CONTENT SUMMARY:
{lecture_body[:10000]}
"""
    try:
        raw_html = call_gemini(html_prompt, max_tokens=16384, temperature=0.2)
        match = re.search(r'(<!DOCTYPE html[\s\S]*?</html>)', raw_html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if "<html" in raw_html.lower():
            start_idx = raw_html.lower().find("<html")
            end_idx = raw_html.lower().rfind("</html>")
            if start_idx != -1 and end_idx != -1:
                return "<!DOCTYPE html>\n" + raw_html[start_idx:end_idx + 7].strip()
            return raw_html.strip()
    except Exception as e:
        print(f"WARNING HTML generation failed: {e}")
    return None


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

    # فصل العنوان الأصلي عن الترجمة الإنجليزية إذا وُجدت بين قوسين
    title_match = re.search(r'\(([^)]+)\)', title)
    if title_match:
        english_title = title_match.group(1).strip()
        latin_title = title[:title_match.start()].strip()
    else:
        english_title = title
        latin_title = title

    prompt = f"""
أنت أستاذ فيزياء ومؤرخ علوم تشرح الأوراق العلمية التأسيسية بأسلوب أكاديمي سلس ومباشر، لطالب فيزياء يريد فهم خطوات البرهان بعمق وبساطة دون تعقيد لغوي.

بيانات العمل العلمي:
- الرمز الأرشيفي: {paper_id}
- التاريخ: {date}
- العالم: {scientist}
- العنوان الأصلي: {latin_title}
- الترجمة الإنجليزية للعنوان: {english_title}
- جهة النشر: {venue}
- الإسهام المحدد: {puzzle}
- الرابط السببي: {bridge}
- الرابط الأرشيفي: {link}

**قواعد الأسلوب والمحتوى:**
1. اكتب بلغة عربية علمية طبيعية ومباشرة تشرح الفكرة الفيزيائية بوضوح دون تقعر أو كلمات مهجورة.
2. لا تستخدم أي رموز تعبيرية (emoji).
3. لا تضع جدولاً منفصلاً للرموز؛ عرّف كل رمز فور ظهوره بسلاسة.
4. التزم حرفياً بأدوات عصر المؤلف ({date}).
5. صمم رسمتين أو ثلاث رسمات توضيحية باستخدام علامات [[FIG_1]] و [[FIG_2]] متبوعة ببلوك كود بايثون ` ```python ... ``` ` لرسم الشكل وحفظه بـ `plt.savefig('fig.png')`.
6. عنوان قسم البرهان يجب أن يكون: `### البرهان` (فقط كلمة البرهان دون أي جملة طويلة بعدها).

**قسّم الشرح وفق هذه العناوين الثمانية بالتسلسل:**
1. `### هوية العمل وسياقه التاريخي` (اكتب في أول سطرين عنوان العمل بالعربية والإنجليزية، ثم المشكلة التاريخية وحلها).
2. `### صوت المؤلف` (اقتباس مترجم ومحدد من النص الأصلي).
3. `### الفروض والأسس التي انطلق منها المؤلف` (مع الرسمة الأولى [[FIG_1]]).
4. `### البرهان` (البرهان المفصل خطوة بخطوة مع الرسمة الثانية [[FIG_2]]).
5. `### التطبيق العملي والأرقام` (حسابات تطبيقية مع رسمة ثالثة [[FIG_3]] إن لزم).
6. `### العدسة الحديثة` (3 إلى 5 أسطر فقط لصياغة النتيجة بأدوات العصر الحالي).
7. `### الثغرة التي لم يرها المؤلف` (نقطة الضعف التاريخية ومن كشفها).
8. `### السؤال المعلَّق` (سؤال ختامي يربط بالورقة القادمة: {bridge}).

**قواعد المعادلات:**
- ضع كل معادلة سطرية بين `$ ... $` وكل معادلة مستقلة بين `$$ ... $$` مع سطر فارغ قبلها وبعدها.
- لا تضع أي كلمة عربية داخل علامات `$ ... $` إطلاقاً.
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
    html_path = f"lectures_html/{base_name}.html"

    # استخراج وتنفيذ جميع الرسومات التوضيحية في سياقها
    raw_lecture_body, generated_figs = process_multiple_diagrams(raw_response, base_name)

    # توحيد عنوان البرهان ليكون '### البرهان' في النص الخام أيضاً
    raw_lecture_body = re.sub(r'###\s*البرهان خطوة بخطوة بأدوات عصر المؤلف', r'### البرهان', raw_lecture_body)

    # توليد ملف الـ HTML التفاعلي
    interactive_html = generate_interactive_html(paper_id, date, scientist, title, raw_lecture_body)
    has_html = False
    if interactive_html and len(interactive_html) > 200:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(interactive_html)
        has_html = True
        print(f"Generated Interactive HTML: {html_path}")
    else:
        print("WARNING Interactive HTML was not generated.")

    # تجهيز النص لـ XeLaTeX أولاً ثم حقن كود الصور في أماكن [[FIG_X]]
    clean_latex_body = sanitize_markdown_for_xelatex(raw_lecture_body)

    for fig_id, img_path in generated_figs.items():
        fig_num = fig_id.split("_")[1]
        latex_fig = (
            f"\n\\begin{{figure}}[H]\n"
            f"\\centering\n"
            f"\\includegraphics[width=0.86\\textwidth]{{{img_path}}}\n"
            f"\\caption{{Figure {fig_num} ({paper_id})}}\n"
            f"\\end{{figure}}\n"
        )
        clean_latex_body = clean_latex_body.replace(f"[[{fig_id}]]", latex_fig)

    clean_latex_body = re.sub(r'\[\[FIG_\d+\]\]', '', clean_latex_body)

    # معالجة عنوان الغلاف بطريقة سليمة تمنع تشويه اللغات المختلطة في XeLaTeX
    safe_title = title.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_scientist = scientist.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_venue = venue.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_puzzle = puzzle.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_bridge = bridge.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    header_block = f"""---
id: "{paper_id}"
date: "{date}"
scientist: "{scientist}"
title: "{title}"
---

\\begin{{center}}
\\Large\\textbf{{{safe_title}}}
\\end{{center}}

\\vspace{{0.5em}}

* **المؤلف:** \\textenglish{{{safe_scientist}}} | **التاريخ:** {date}
* **جهة النشر:** \\textenglish{{{safe_venue}}}
* **ملخص الإسهام:** \\textenglish{{{safe_puzzle}}}
* **الأثر التاريخي:** \\textenglish{{{safe_bridge}}}
* **الرابط الأرشيفي:** <{link}>

---

{clean_latex_body}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header_block)

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

    # حفظ نسخة Obsidian
    obsidian_body = raw_lecture_body
    for fig_id, img_path in generated_figs.items():
        fig_num = fig_id.split("_")[1]
        obsidian_body = obsidian_body.replace(f"[[{fig_id}]]", f"\n![Figure {fig_num}](../{img_path})\n")
    obsidian_body = re.sub(r'\[\[FIG_\d+\]\]', '', obsidian_body)

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
* **ملخص الإسهام:** {puzzle}
* **الأثر التاريخي:** {bridge}
* **الرابط الأرشيفي:** <{link}>

---

{obsidian_body}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(obsidian_md)

    # استخراج مقدمة مبسطة ومفيدة من المحتوى لإدراجها في نص الإيميل
    intro_match = re.search(r'###\s*هوية العمل وسياقه التاريخي\s*\n\n(.*?)(?=\n\n###|\Z)', raw_lecture_body, re.DOTALL)
    if intro_match:
        intro_raw = intro_match.group(1).strip()
        intro_sentences = [s.strip() for s in re.split(r'[\.\n]', intro_raw) if len(s.strip()) > 15]
        email_intro = "\n\n".join(intro_sentences[:2])
    else:
        email_intro = puzzle

    msg = EmailMessage()
    msg["Subject"] = f"[{paper_id} | {date}] {scientist.split('(')[0].strip()}: محاضرة تأسيسية ومحاكاة"
    msg["From"] = f"Physics Archive <{GMAIL_USER}>"
    msg["To"] = SUBSCRIBERS[0]
    if len(SUBSCRIBERS) > 1:
        msg["Bcc"] = ", ".join(SUBSCRIBERS[1:])

    html_status_text = "3. ملف HTML تفاعلي (محاكاة الحركة والبارامترات)." if has_html else ""

    email_body = f"""[{paper_id}] {scientist.split('(')[0].strip()} -- {date}

العمل: {title}

نظرة عامة ومقدمة المحاضرة:
----------------------------------------
{email_intro}

الإسهام الفيزيائي:
{puzzle}

الأثر التاريخي:
{bridge}

المصدر الأرشيفي:
{link}

----------------------------------------
المرفقات:
1. نسخة PDF مطبوعة ومجهزة بالرسومات الهندسية.
2. نسخة Markdown كاملة لـ Obsidian.
{html_status_text}
"""
    msg.set_content(email_body.strip())

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
    print("Email sent successfully.")

    df.at[idx, "Status"] = "SENT"
    df.at[idx, "Sent_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
