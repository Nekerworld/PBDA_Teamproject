"""Rasterize the final PDF for mandatory visual review."""
import argparse,json
from pathlib import Path
import pypdfium2 as pdfium
from pypdf import PdfReader


def main(pdf,output):
    pdf=Path(pdf);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    doc=pdfium.PdfDocument(str(pdf));pages=[]
    for i in range(len(doc)):
        image=doc[i].render(scale=2.).to_pil()
        path=out/f'page_{i+1:02}.png';image.save(path)
        pages.append({'page':i+1,'image':str(path),'size':list(image.size)})
    reader=PdfReader(pdf)
    texts=[p.extract_text() or '' for p in reader.pages]
    if not all(t.strip() for t in texts):raise ValueError('Empty text page in generated report')
    result={'pages':len(pages),'images':pages,'text_characters':[len(t) for t in texts],
        'visual_review':'pending human/assistant image inspection'}
    (out/'render_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pdf',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.pdf,a.output)
