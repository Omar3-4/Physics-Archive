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
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 32768}
        }
        res = requests.post(url, json=payload, timeout=240)
        if res.status_code == 200:
            data = res.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "\n".join(p["text"] for p in parts if "text" in p and not p.get("thought", False)).strip()
            if len(text) > 100:
                print(f"Generated via {model} ({len(text)} chars)")
                return text
        else:
            print(f"WARNING {model} failed with HTTP {res.status_code}")
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
        # إزالة كود البايثون من النص مع الإبقاء على علامة [[FIG_X]] في مكانها وسط الفقرات
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

    # تنظيف أي كود بايثون متبقٍ
    clean_text = re.sub(r'```python[\s\S]*?```', '', clean_text).strip()
    return clean_text, generated_figs


def sanitize_markdown_for_xelatex(text):
    # 1. توحيد صيغ الرياضيات
    text = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'([^\n])\n\$\$', r'\1\n\n$$', text)
    text = re.sub(r'\$\$\n([^\n])', r'$$\n\n\1', text)

    # 2. إزالة أي أوامر \textenglish مكسورة قد يكتبها النموذج تلقائياً
    text = re.sub(r'\\textenglish\{([^}]*)\}', r'\1', text)

    # 3. حماية النصوص خارج المعادلات وحماية علامات [[FIG_X]]
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$|\[\[FIG_\d+\]\])', text)

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

    for i in range(len(parts)):
        part = parts[i]
        if part.startswith("$") or part.startswith("[[FIG_"):
            continue
        part = math_cmds.sub(r'$\1$', part)
        part = re.sub(r'(?<!\\)_', r'\\_', part)
        parts[i] = part

    return "".join(parts)


def generate_interactive_html(paper_id, date, scientist, title, lecture_body):
    html_prompt = f"""
You are an expert computational physicist and scientific visualization engineer.
Based on the historical physics lecture below for:
- Paper: {title} ({date}) by {scientist}

Build a **single-file, self-contained, interactive HTML5 Canvas demonstration** that visually animates and demonstrates the exact mathematical/geometric derivation step-by-step using ONLY period-accurate tools from {date}.

**CRITICAL TECHNICAL & DESIGN RULES:**
1. **KaTeX Auto-Render Mandatory:** Include KaTeX CSS, JS, and the `auto-render` extension (`https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js`). Call `renderMathInElement(document.body, {{ delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}] }});` on page load AND whenever dynamic DOM elements are updated so NO raw `$` signs EVER appear on screen!
2. **Clear Visual Geometry + Concise Arabic Intuition:**
   - Left/Main panel: Dark-mode HTML5 `<canvas>` (`#090d16`) drawing the exact geometric construction with clear labels, colored vectors/angles, dashed auxiliary lines, and interactive sliders (angles, lengths, time) + Play/Pause & Step buttons.
   - Right panel: Step-by-step derivation cards. Each card must show the symbolic equation (rendered via KaTeX) accompanied by a **single short, crystal-clear Arabic sentence** explaining what happens geometrically in that step (so a reader weak in geometry immediately sees why the step works). Clicking a step highlights the exact lines/angles/triangles on the canvas.
3. **Historical Fidelity:** Use ONLY the exact symbols and geometric methods from {date}.
4. **Output Format:** Return ONLY valid HTML starting with `<!DOCTYPE html>` and ending with `</html>`. No markdown fences.

**LECTURE CONTENT:**
{lecture_body[:16000]}
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
أنت أستاذ فيزياء ومؤرخ علوم تشرح الأوراق العلمية التأسيسية بأسلوب أكاديمي سلس وواضح ومباشر، لطالب فيزياء يريد أن يعيش في عصر المؤلف ويفهم كل خطوة بعمق وبساطة دون تعقيد لغوي.

بيانات العمل العلمي:
- الرمز الأرشيفي: {paper_id}
- التاريخ: {date}
- العالم: {scientist}
- العنوان: {title}
- جهة النشر: {venue}
- الإسهام المحدد: {puzzle}
- الرابط السببي: {bridge}
- الرابط الأرشيفي: {link}

**قواعد صارمة جداً للأسلوب والمحتوى (إلزامية):**
1. **لغة عربية واضحة ومباشرة:**
   - اكتب بلغة عربية علمية طبيعية ومباشرة تشرح الفكرة الفيزيائية بوضوح تام دون تقعر لفظي أو كلمات مهجورة.
   - لا تستخدم أي رموز تعبيرية (emoji) نهائياً.
   - لا تكتب مقدمات ترحيبية أو عبارات مبالغة وصفية، وابدأ العرض مباشرة.
   - لا تكتب اقتباسات باللغات الأجنبية دون ترجمة. اكتب المصطلح الإنجليزي الأساسي بين قوسين عاديين عند الحاجة فقط دون أوامر معقدة.
2. **تعريف كل رمز في سياقه الطبيعي:**
   - لا تضع جدولاً أو قائمة منفصلة للرموز في البداية.
   - كلما ظهر رمز رياضي أو هندسي لأول مرة في المعادلة أو البرهان، عرّفه فوراً داخل نفس الجملة بسلاسة ووضوح.
3. **التقيد الحرفي بأدوات عصر المؤلف ({date}):**
   - اشرح البرهان بنفس الأدوات الرياضية والهندسية التي استخدمها المؤلف في عصره فقط (مثل الهندسة الإقليدية، تشابه المثلثات، النسب، الأوتار)، ويُمنع استخدام أي أداة رياضية لم تكن قد اختُرعت بعد في تاريخ {date}.
4. **رسومات توضيحية متعددة وكل رسمة في سياقها داخل النص:**
   - صمّم رسمتين أو ثلاث رسمات هندسية/فيزيائية مختلفة توضح مراحل الفكرة (مثلاً: الرسمة الأولى للنموذج القديم مقابل الجديد أو المشكلة الفيزيائية، والرسمة الثانية للبناء الهندسي للبرهان، والرسمة الثالثة للتطبيق أو مقارنة النتائج).
   - ضع كل رسمة في موضعها الطبيعي داخل الفقرة التي تشرحها باستخدام العلامة `[[FIG_1]]` متبوعة مباشرة ببلوك كود بايثون ` ```python ... ``` ` يرسمها، ثم `[[FIG_2]]` متبوعة بكودها في موضعها المناسب، وهكذا.
   - تأكد في كل كود بايثون أن النصوص والتسميات داخل الرسمة بالإنجليزية فقط (`O, M, L, theta`)، وأن الخطوط واضحة ومقروءة، وتُحفظ بـ `plt.savefig('fig.png')`.

**قسّم الشرح وفق هذه المحاور بالتسلسل:**
1. **هوية العمل وسياقه التاريخي:** (عنوان العمل مترجماً للعربية والإنجليزية، والمشكلة الفيزيائية أو الفلكية التي كانت تؤرق العلماء قبل {date} وكيف حلها المؤلف).
2. **صوت المؤلف:** (اقتبس بالترجمة العربية المباشرة جملة أو فقرة واحدة من النص الأصلي للمؤلف مع ذكر الكتاب والفصل أو الصفحة لربط القارئ بالوثيقة الأصلية).
3. **الفروض والأسس التي انطلق منها المؤلف:** (شرح الفرضيات الفيزيائية والهندسية الأساسية بلغة واضحة، مع وضع الرسمة التوضيحية الأولى `[[FIG_1]]` في سياقها).
4. **البرهان خطوة بخطوة بأدوات عصر المؤلف:** (تتبع البرهان الكامل بتسلسل منطقي؛ اشرح كل خطوة هندسية كأنك ترسمها أمام القارئ، عرّف كل رمز في مكانه، وضع الرسمة التوضيحية الثانية `[[FIG_2]]` وسط خطوات البرهان لتوضيح المثلثات أو الدوائر أو المسارات).
5. **التطبيق العملي والأرقام:** (تطبيق المعادلات على حالة فيزيائية ملموسة بالأرقام كما فعل المؤلف، مع رسمة ثالثة `[[FIG_3]]` إن لزم الأمر).
6. **العدسة الحديثة:** (في 3 إلى 5 أسطر فقط: كيف تُعبّر عن النتيجة الجوهرية ذاتها اليوم بأدوات حساب المتجهات أو التفاضل والتكامل لإبراز الفارق الزمني).
7. **الثغرة التي لم يرها المؤلف:** (نقطة ضعف واحدة أو افتراض محدد في هذه الورقة أثبت العلم لاحقاً خطأه أو محدوديته، مع ذكر من كشفه).
8. **السؤال المعلَّق:** (سؤال واحد مفتوح في الختام يبرز التناقض أو الثغرة التي ستحلها الورقة التالية في السلسلة: {bridge}).

**قواعد المعادلات:**
- ابدأ مباشرة بالمحور الأول دون مقدمات.
- ضع كل معادلة سطرية بين `$ ... $` وكل معادلة مستقلة بين `$$ ... $$` مع ترك سطر فارغ قبلها وبعدها.
- لا تضع أي كلمة عربية داخل علامات `$ ... $` أو `$$ ... $$` إطلاقاً.
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

    # توليد ملف الـ HTML التفاعلي
    interactive_html = generate_interactive_html(paper_id, date, scientist, title, raw_lecture_body)
    has_html = False
    if interactive_html:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(interactive_html)
        has_html = True
        print(f"Generated Interactive HTML: {html_path}")

    # تجهيز النص لـ XeLaTeX أولاً ثم حقن كود الصور النظيف في أماكن [[FIG_X]]
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

    # إزالة أي علامة [[FIG_X]] فشل رسمها حتى لا تظهر في النص
    clean_latex_body = re.sub(r'\[\[FIG_\d+\]\]', '', clean_latex_body)

    safe_title = title.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_scientist = scientist.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_venue = venue.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_puzzle = puzzle.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    safe_bridge = bridge.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    full_markdown_for_pdf = f"""---
id: "{paper_id}"
date: "{date}"
scientist: "{scientist}"
title: "{title}"
---

# \\textenglish{{{safe_title}}}

* **المؤلف:** \\textenglish{{{safe_scientist}}} | **التاريخ:** {date}
* **جهة النشر:** \\textenglish{{{safe_venue}}}
* **ملخص الإسهام:** \\textenglish{{{safe_puzzle}}}
* **الأثر التاريخي:** \\textenglish{{{safe_bridge}}}
* **الرابط الأرشيفي:** <{link}>

---

{clean_latex_body}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown_for_pdf)

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

    # حفظ نسخة Obsidian النظيفة بالصور في مواضعها
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

    msg = EmailMessage()
    msg["Subject"] = f"[{paper_id} | {date}] {scientist.split('(')[0].strip()}: محاضرة تأسيسية + محاكاة تفاعلية"
    msg["From"] = f"Physics Archive <{GMAIL_USER}>"
    msg["To"] = SUBSCRIBERS[0]
    if len(SUBSCRIBERS) > 1:
        msg["Bcc"] = ", ".join(SUBSCRIBERS[1:])

    msg.set_content(
        f"[{paper_id}] {scientist.split('(')[0].strip()} -- {date}\n\n"
        f"{title}\n\n"
        f"ملخص الإسهام: {puzzle}\n\n"
        f"الأثر التاريخي والرابط السببي: {bridge}\n\n"
        f"المصدر الأصلي: {link}\n\n"
        f"المرفقات:\n"
        f"1. نسخة PDF مطبوعة بمحرك XeLaTeX مع الرسومات التوضيحية السياقية.\n"
        f"2. نسخة Markdown لـ Obsidian.\n"
        f"3. ملف HTML تفاعلي متقدم بالأنيميشن وتحكم البارامترات."
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
    print("Email sent successfully.")

    df.at[idx, "Status"] = "SENT"
    df.at[idx, "Sent_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
