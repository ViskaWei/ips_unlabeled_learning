#!/usr/bin/env python3
"""
Markdown preview server with syntax highlighting and LaTeX support.
Usage: python3 md_server.py [port] [root_dir]
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import sys
from pathlib import Path
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
import urllib.parse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 6419
ROOT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/swei20/AI_Research_Template")

# CSS for syntax highlighting (GitHub Light theme)
CSS = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 40px;
    background: #fff;
    color: #24292e;
    line-height: 1.6;
}
h1, h2, h3, h4 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
pre { background: #f6f8fa; border-radius: 6px; padding: 16px; overflow-x: auto; border: 1px solid #e1e4e8; }
code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 14px; }
p code, li code { background: #f6f8fa; padding: 2px 6px; border-radius: 3px; color: #d73a49; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th, td { border: 1px solid #dfe2e5; padding: 8px 12px; text-align: left; }
th { background: #f6f8fa; }
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
blockquote { border-left: 4px solid #dfe2e5; margin: 0; padding-left: 16px; color: #6a737d; }
.nav { background: #f6f8fa; padding: 10px 16px; margin: -20px -40px 20px; border-bottom: 1px solid #eaecef; }
.nav a { margin-right: 16px; }
/* 正文的列表样式 */
ul { padding-left: 24px; margin: 8px 0; list-style-type: disc; }
ul li { padding: 2px 0; }
ol { padding-left: 24px; margin: 8px 0; }
ol li { padding: 2px 0; }
/* 目录导航的列表样式 */
ul.dir-listing { list-style-type: none; padding-left: 0; }
ul.dir-listing li { padding: 4px 0; }
/* 图片样式 - 限制为原始大小的 1/3 */
img { max-width: 600px; width: 33.33%; height: auto; display: block; margin: 10px 0; }

/* Syntax highlighting - Colorful Light theme */
.codehilite { background: #f6f8fa; color: #24292e; }
.codehilite pre { margin: 0; background: transparent; border: none; padding: 0; }
.codehilite .hll { background-color: #fffbdd }
.codehilite .c { color: #43a047; font-style: italic } /* Comment - 亮绿 */
.codehilite .k { color: #0277bd; font-weight: bold } /* Keyword - 天蓝 */
.codehilite .o { color: #d84315 } /* Operator - 深橙红 */
.codehilite .p { color: #546e7a } /* Punctuation - 蓝灰 */
.codehilite .ch { color: #43a047; font-style: italic } /* Comment.Hashbang - 亮绿 */
.codehilite .cm { color: #43a047; font-style: italic } /* Comment.Multiline - 亮绿 */
.codehilite .c1 { color: #43a047; font-style: italic } /* Comment.Single - 亮绿 */
.codehilite .cs { color: #43a047; font-style: italic; font-weight: bold } /* Comment.Special - 亮绿粗体 */
.codehilite .gd { color: #c62828; background-color: #ffebee } /* Generic.Deleted - 红 */
.codehilite .gi { color: #2e7d32; background-color: #e8f5e9 } /* Generic.Inserted - 绿 */
.codehilite .kc { color: #6a1b9a } /* Keyword.Constant (True/False/None) - 深紫 */
.codehilite .kd { color: #1565c0; font-weight: bold } /* Keyword.Declaration (def/class) - 深蓝 */
.codehilite .kn { color: #00838f } /* Keyword.Namespace (import/from) - 青色 */
.codehilite .kp { color: #1565c0 } /* Keyword.Pseudo - 深蓝 */
.codehilite .kr { color: #1565c0; font-weight: bold } /* Keyword.Reserved - 深蓝 */
.codehilite .kt { color: #00695c } /* Keyword.Type - 墨绿 */
.codehilite .m { color: #ab47bc } /* Literal.Number - 更亮紫 */
.codehilite .s { color: #c62828 } /* Literal.String - 红色 */
.codehilite .na { color: #00838f } /* Name.Attribute - 青色 */
.codehilite .nb { color: #43a047 } /* Name.Builtin (len/range/print) - 亮绿 */
.codehilite .nc { color: #d84315; font-weight: bold } /* Name.Class - 深橙 */
.codehilite .no { color: #6a1b9a } /* Name.Constant - 深紫 */
.codehilite .nd { color: #7b1fa2 } /* Name.Decorator (@) - 紫色 */
.codehilite .ne { color: #c62828; font-weight: bold } /* Name.Exception - 红色 */
.codehilite .nf { color: #d32f2f; font-weight: bold } /* Name.Function - 红色 */
.codehilite .nn { color: #00838f } /* Name.Namespace - 青色 */
.codehilite .nt { color: #00695c } /* Name.Tag - 墨绿 */
.codehilite .nv { color: #bf360c } /* Name.Variable - 砖红 */
.codehilite .ow { color: #1565c0; font-weight: bold } /* Operator.Word (and/or/in) - 深蓝 */
.codehilite .w { color: #24292e } /* Text.Whitespace */
.codehilite .mb { color: #ab47bc } /* Literal.Number.Bin - 更亮紫 */
.codehilite .mf { color: #e91e63 } /* Literal.Number.Float - 粉红 */
.codehilite .mh { color: #ba68c8 } /* Literal.Number.Hex - 浅紫 */
.codehilite .mi { color: #ab47bc } /* Literal.Number.Integer - 更亮紫 */
.codehilite .mo { color: #ba68c8 } /* Literal.Number.Oct - 浅紫 */
.codehilite .sa { color: #ef6c00 } /* Literal.String.Affix (f/r/b) - 橙色 */
.codehilite .sb { color: #c62828 } /* Literal.String.Backtick - 红色 */
.codehilite .sc { color: #ad1457 } /* Literal.String.Char - 玫红 */
.codehilite .dl { color: #c62828 } /* Literal.String.Delimiter - 红色 */
.codehilite .sd { color: #2e7d32; font-style: italic } /* Literal.String.Doc (docstring) - 森林绿 */
.codehilite .s2 { color: #c62828 } /* Literal.String.Double - 红色 */
.codehilite .se { color: #e65100 } /* Literal.String.Escape (\n) - 亮橙 */
.codehilite .sh { color: #c62828 } /* Literal.String.Heredoc - 红色 */
.codehilite .si { color: #7b1fa2 } /* Literal.String.Interpol ({}) - 紫色 */
.codehilite .sx { color: #c62828 } /* Literal.String.Other - 红色 */
.codehilite .sr { color: #00838f } /* Literal.String.Regex - 青色 */
.codehilite .s1 { color: #c62828 } /* Literal.String.Single - 红色 */
.codehilite .ss { color: #7b1fa2 } /* Literal.String.Symbol - 紫色 */
.codehilite .bp { color: #6a1b9a } /* Name.Builtin.Pseudo (self/cls) - 深紫 */
.codehilite .fm { color: #d32f2f; font-weight: bold } /* Name.Function.Magic (__init__) - 红色 */
.codehilite .vc { color: #bf360c } /* Name.Variable.Class - 砖红 */
.codehilite .vg { color: #e65100 } /* Name.Variable.Global - 亮橙 */
.codehilite .vi { color: #bf360c } /* Name.Variable.Instance - 砖红 */
.codehilite .il { color: #ab47bc } /* Literal.Number.Integer.Long - 更亮紫 */
</style>
"""

def render_markdown(content):
    md = markdown.Markdown(extensions=[
        CodeHiliteExtension(css_class='codehilite', guess_lang=True, linenums=False),
        FencedCodeExtension(),
        TableExtension(),
    ])
    return md.convert(content)

def make_page(title, nav_html, body_html):
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    {CSS}
    <!-- MathJax for LaTeX rendering -->
    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }},
        svg: {{
            fontCache: 'global'
        }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
{nav_html}
{body_html}
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def do_GET(self):
        try:
            path = urllib.parse.unquote(self.path.split('?')[0])
            if path == '/':
                path = '/'
            
            file_path = ROOT_DIR / path.lstrip('/')
            
            # Navigation
            rel_path = file_path.relative_to(ROOT_DIR) if file_path != ROOT_DIR else Path('.')
            nav = '<div class="nav"><a href="/">🏠 Home</a>'
            if rel_path != Path('.') and rel_path.parent != Path('.'):
                nav += f'<a href="/{rel_path.parent}/">📁 {rel_path.parent}</a>'
            nav += '</div>'
            
            if file_path.suffix.lower() == '.md' and file_path.exists():
                # Render markdown
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                html_body = render_markdown(content)
                html = make_page(rel_path.name, nav, html_body)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', len(html.encode('utf-8')))
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
                
            elif file_path.is_dir():
                # List directory
                items = sorted(file_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                links = []
                for item in items:
                    if item.name.startswith('.'):
                        continue
                    item_rel = item.relative_to(ROOT_DIR)
                    icon = '📁' if item.is_dir() else ('📄' if item.suffix == '.md' else '📋')
                    href = f'/{item_rel}{"/" if item.is_dir() else ""}'
                    links.append(f'<li>{icon} <a href="{href}">{item.name}</a></li>')
                
                body = f'<h1>📁 {rel_path if rel_path != Path(".") else "AI_Research_Template"}</h1><ul class="dir-listing">{"".join(links)}</ul>'
                html = make_page(str(rel_path), nav, body)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', len(html.encode('utf-8')))
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            else:
                # 404
                self.send_response(404)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(b'<h1>404 Not Found</h1>')
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.send_response(500)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(f'<h1>500 Error</h1><pre>{e}</pre>'.encode('utf-8'))

if __name__ == '__main__':
    os.chdir(ROOT_DIR)
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    server.allow_reuse_address = True
    print(f"🌐 Markdown server running at http://localhost:{PORT}/")
    print(f"📁 Serving: {ROOT_DIR}")
    server.serve_forever()
