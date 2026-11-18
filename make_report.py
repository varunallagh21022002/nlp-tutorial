"""Build a submission PDF from the actual captured execution results."""
import json
import math
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT/'outputs/results.json').read_text())
ENV = json.loads((ROOT/'outputs/environment.json').read_text())
OUT = ROOT/'output/pdf/NLP_Tutorial_Practical_Record.pdf'
W,H=A4
M=48
INK=HexColor('#18304A'); TEAL=HexColor('#087D8B'); GRAY=HexColor('#506070')
BODY=ParagraphStyle('body',fontName='Helvetica',fontSize=10.5,leading=15,textColor=INK)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=9,leading=12)
TITLE=ParagraphStyle('title',parent=BODY,fontName='Helvetica-Bold',fontSize=22,leading=27)
c=canvas.Canvas(str(OUT),pagesize=A4)
c.setTitle('NLP Tutorial - 15 ML Practicals - Varun Allagh')
c.setAuthor('Varun Allagh')
page=0

def new_page(label):
    global page
    if page: c.showPage()
    page+=1
    c.setFillColor(INK); c.rect(0,H-29,W,29,fill=1,stroke=0)
    c.setFillColor(white); c.setFont('Helvetica-Bold',8)
    c.drawString(M,H-19,'NLP TUTORIAL  /  PRACTICAL RECORD')
    c.setFillColor(GRAY); c.setFont('Helvetica',8)
    c.drawString(M,25,label)
    c.drawRightString(W-M,25,f'{page:02d}')
    c.setStrokeColor(HexColor('#CCD8E0')); c.line(M,40,W-M,40)
    return H-65

def para(text,y,style=BODY):
    p=Paragraph(text,style); _,height=p.wrap(W-2*M,H)
    p.drawOn(c,M,y-height)
    return y-height-12

def heading(text,y):
    c.setFillColor(TEAL); c.setFont('Helvetica-Bold',12)
    c.drawString(M,y,text)
    return y-18

def code(lines,y,size=8.4,leading=11.1):
    c.setFillColor(HexColor('#F1F5F8'))
    c.rect(M-6,y-len(lines)*leading-8,W-2*M+12,len(lines)*leading+14,fill=1,stroke=0)
    c.setFillColor(INK); c.setFont('Courier',size)
    for line in lines:
        c.drawString(M,y,line); y-=leading
    return y-18

def wrap_code(source,width=92):
    lines=[]
    for line in source.splitlines():
        # Display wrapping only; executable source remains in notebooks/scripts.
        lines.extend(textwrap.wrap(line,width=width,replace_whitespace=False,
            drop_whitespace=False,subsequent_indent='    ') or [''])
    return lines

# Cover
new_page('Submission record')
c.setFillColor(TEAL); c.rect(M,H-165,55,5,fill=1,stroke=0)
y=H-195
y=para('NLP TUTORIAL',y,ParagraphStyle('cover',parent=TITLE,fontSize=34,leading=40))
y=para('Machine Learning<br/>Practical Record',y,ParagraphStyle('subtitle',parent=TITLE,fontSize=25,leading=32))
y-=20
y=para('15 implemented practicals<br/>Python and Google Colab',y)
y-=25
y=heading('Submitted by',y); y=para('Varun Allagh',y)
y=heading('Date',y); y=para('18 November 2026',y)
y-=20
y=para('Based on the supplied <i>15 ML Tutorial Problem Statements.pdf</i>. '
    'All prescribed models are implemented in the original assignment order. '
    'Every practical includes a dedicated output page.',y)
y=para('Repository: <link href="https://github.com/varunallagh21022002/nlp-tutorial" color="#087D8B">github.com/varunallagh21022002/nlp-tutorial</link>',y,SMALL)
y=para('Notebook: <link href="https://colab.research.google.com/github/varunallagh21022002/nlp-tutorial/blob/main/All_15_Practicals_Colab.ipynb" color="#087D8B">Open all 15 practicals in Google Colab</link>',y,SMALL)
# Index
Y=new_page('Index')
Y=para('Index',Y,TITLE)
Y=para('Each practical spans three pages: method, implementation and a separate output page.',Y)
c.setFillColor(INK);c.setFont('Helvetica-Bold',9)
c.drawString(M,Y,'No.'); c.drawString(M+30,Y,'Practical');c.drawRightString(W-M,Y,'Pages');Y-=22
for p in RESULTS:
    start=4+(p['number']-1)*3
    c.setFillColor(INK);c.setFont('Helvetica',9.2)
    c.drawString(M,Y,str(p['number']).zfill(2))
    title=p['title']
    wrapped=textwrap.wrap(title,65)
    for line in wrapped:
        c.drawString(M+30,Y,line);Y-=13
    c.drawRightString(W-M,Y+13,f'{start}-{start+2}')
    Y-=12
Y-=10
Y=para('Shared setup and evaluation functions follow the practicals. These functions are embedded in every standalone notebook and printed in full in the appendix.',Y,SMALL)
# Execution and provenance
Y=new_page('Execution and data provenance'); Y=para('Execution and reproducibility',Y,TITLE)
Y=heading('How to run',Y)
Y=para('Open the linked combined notebook in Google Colab, select a CPU runtime, and choose Runtime &gt; Run all. '
    'Text datasets are embedded in the notebooks; no private Drive files or external dataset downloads are required.',Y)
Y=heading('Data provenance',Y)
Y=para('Iris, breast cancer and digits use the packaged scikit-learn benchmark datasets. '
    'Other numerical datasets are generated with documented rules and random seed 42. '
    'The SMS, review and news corpora contain 80, 80 and 100 original teaching examples. '
    'Generated data and authored text are explicitly identified in each practical.',Y)
Y=heading('Evaluation protocol',Y)
Y=para('Supervised models use a 75/25 split, stratified for classification. Transformations are fitted on training data only. '
    'Sales uses chronological splitting. Isolation Forest receives no labels during fitting. '
    'K-Means uses internal silhouette scores; PCA reports held-out reconstruction error. '
    'Model settings are fixed in advance; no test-label hyperparameter search is performed.',Y)
Y=heading('Recorded execution',Y)
provenance='The printed metrics, text output and charts were captured by executing the supplied Python code locally, with independent state for every practical.'
if (ROOT/'outputs/colab_verification.json').exists():
    verify=json.loads((ROOT/'outputs/colab_verification.json').read_text())
    provenance+=' The combined notebook was also run successfully in Google Colab; the final completion marker and absence of cell errors were verified.'
Y=para(provenance,Y)
Y=para('Local environment: '+', '.join(f'{k} {ENV[k]}' for k in ['python','numpy','pandas','scikit-learn','matplotlib'])+'. Seed: 42. '
    'Compatible dependency ranges are installed by the first Colab cell; exact local versions are recorded in the repository.',Y,SMALL)
Y=heading('Interpretation',Y)
Y=para('Metrics on generated or small authored datasets demonstrate implementation correctness under the stated setup. '
    'They are not evidence of deployment readiness. A single held-out split does not provide a confidence interval.',Y)
for p in RESULTS:
    n=p['number']; label=f'Practical {n:02d} / {p["title"]}'
    Y=new_page(label); Y=para(f'Practical {n:02d}<br/>{escape(p["title"])}',Y,TITLE)
    for title,txt in [('Aim',p['aim']),('Model',p['model']),('Dataset and features',p['data']),('Theory',p['theory'])]:
        Y=heading(title,Y);Y=para(escape(txt),Y)
    Y=heading('Algorithm',Y)
    if n==8: steps=['Generate feature-only customer records and standardize the three features.','Fit K-Means for k from 2 to 6; choose the highest silhouette score.','Inspect cluster sizes and inverse-transformed centres; plot income against spending.']
    elif n==13: steps=['Load digit pixels and split into training and test images.','Fit a 95-percent-variance PCA and a separate two-component PCA on training images.','Measure test reconstruction error and visualize the two-dimensional test projection.']
    elif n==10: steps=['Generate transactions and create a stratified held-out split.','Fit Isolation Forest without labels; convert negative predictions to anomaly labels.','Report per-class metrics and average precision; plot the confusion matrix.']
    else: steps=['Load or generate the stated data and define features and targets.','Split training and test data; fit required preprocessing inside the model pipeline.' if n!=15 else 'Use the first 450 days for training and the last 150 for rolling next-day evaluation.','Fit the prescribed model and predict held-out observations.','Compare with a baseline and inspect metrics, example predictions and the chart.']
    for i,s in enumerate(steps,1):Y=para(f'{i}. {escape(s)}',Y,SMALL)
    Y=para('Notebook: '+escape(p['filename']),Y,SMALL)
    # Implementation is complete once shared setup in appendix has been run.
    Y=new_page(label+' / Implementation');Y=para('Implementation',Y,TITLE)
    Y=para('Run the shared imports and evaluation functions in the appendix first. '
        'Every Colab notebook already includes this setup. Text datasets are embedded in notebook setup cells and available as CSV files in data/.',Y,SMALL)
    source=p['code']
    if n in (7,12,14):source="DATA = Path('data')\n"+source
    lines=wrap_code(source)
    assert len(lines)<=49,(n,len(lines))
    Y=code(lines,Y)
    Y=heading('Implementation note',Y)
    Y=para('The full runnable source is included in the notebook and matching Python script. '
        'The shared evaluate function prints metrics and saves the plot shown on the next page.',Y,SMALL)
    # Mandatory separate output page
    Y=new_page(label+' / Output');Y=para(f'Output - Practical {n:02d}',Y,TITLE)
    lines=wrap_code(p['output'],width=95)
    Y=code(lines,Y,size=8.3,leading=10)
    # Actual chart, scaled to fit available space and preserve aspect ratio.
    from reportlab.lib.utils import ImageReader
    img=ImageReader(str(ROOT/p['image']));iw,ih=img.getSize()
    maxheight=min(225,Y-165)
    scale=min((W-2*M)/iw,maxheight/ih)
    assert scale>0,(n,Y)
    dw,dh=iw*scale,ih*scale
    c.drawImage(img,(W-dw)/2,Y-dh,width=dw,height=dh)
    Y-=dh+18
    Y=heading('Result and conclusion',Y)
    key,value=next(iter(p['metrics'].items()))
    result=f'The implementation completed successfully. {key.replace("_"," ")} = {value:.4f}. '+p['limitation']
    Y=para(escape(result),Y,SMALL)
    assert Y>45,(n,Y)
# Appendix with complete reusable code
source=(ROOT/'src/common.py').read_text()
lines=wrap_code(source)
chunks=[lines[i:i+55] for i in range(0,len(lines),55)]
for i,chunk in enumerate(chunks):
    Y=new_page('Appendix / Shared Python setup')
    Y=para(f'Shared setup ({i+1}/{len(chunks)})',Y,TITLE)
    Y=code(chunk,Y,size=8,leading=10.8)
Y=new_page('References and submission contents');Y=para('References and files',Y,TITLE)
for title,url in [
    ('scikit-learn: packaged datasets','https://scikit-learn.org/stable/datasets/toy_dataset.html'),
    ('scikit-learn: estimators and preprocessing','https://scikit-learn.org/stable/user_guide.html'),
    ('Google Colab','https://colab.research.google.com/'),
    ('Code and executed notebooks','https://github.com/varunallagh21022002/nlp-tutorial')]:
    Y=heading(title,Y);Y=para(f'<link href="{url}" color="#087D8B">{url}</link>',Y,SMALL)
Y=heading('Assignment source',Y);Y=para('15 ML Tutorial Problem Statements.pdf, supplied by the student. The 15 topics and model choices follow that document; it does not specify datasets or hyperparameters.',Y)
Y=heading('Included files',Y)
Y=para('One submission PDF; one combined Colab notebook; 15 individual Colab notebooks; Python implementations; teaching datasets; measured text outputs; plots; metrics; environment details; and reproducible build scripts.',Y)
Y=heading('Reading the output pages',Y)
Y=para('MAE and RMSE use the target unit. R2 compares regression predictions with a constant-mean reference. '
    'Classification reports show precision, recall and F1 for each label; support is the number of test examples. '
    'Average precision measures ranked anomaly detection. Silhouette evaluates cluster separation. '
    'PCA reconstruction MSE is measured on pixels scaled to the interval from zero to one.',Y)
c.save()
print(f'PDF written: {OUT} ({page} pages)')
