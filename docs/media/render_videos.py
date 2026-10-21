"""Continuous BETA research stories, rendered from reported data and real inputs."""
from pathlib import Path
import argparse,json,math,subprocess,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parent;ASSETS=ROOT/'assets'
W,H,S,FPS,ACTIVE,HOLD=1280,720,1.5,30,20,.5
BG='#100B1C';PANEL='#1E152E';BORDER='#3B2B51';WHITE='#F7F4FC';MUTED='#C2B4D1'
PURPLE='#C27AFF';TEAL='#6BD5C3';RED='#F3919C';GOLD='#E5BD75'
FONT='/System/Library/Fonts/Supplemental/Arial.ttf';BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FC={};IC={};px=lambda x:round(x*S)
def clamp(x):return min(1,max(0,x))
def ease(x):x=clamp(x);return x*x*(3-2*x)
def font(n,b=False):
 k=(n,b)
 if k not in FC:FC[k]=ImageFont.truetype(BOLD if b else FONT,px(n))
 return FC[k]
def text(im,x,y,s,n=24,c=WHITE,b=False,a='lt'):
 ImageDraw.Draw(im).text((px(x),px(y)),str(s),font=font(n,b),fill=c,anchor=a)
def line(im,pts,c=BORDER,w=1):
 ImageDraw.Draw(im).line([(px(x),px(y)) for x,y in pts],fill=c,width=max(1,px(w)),joint='curve')
def rect(im,xy,fill=PANEL,outline=None,r=10,w=1):
 ImageDraw.Draw(im).rounded_rectangle(tuple(px(x) for x in xy),radius=px(r),fill=fill,outline=outline,width=max(1,px(w)))
def dot(im,x,y,r,c):ImageDraw.Draw(im).ellipse((px(x-r),px(y-r),px(x+r),px(y+r)),fill=c)
def arrow(im,pts,c=PURPLE,w=2,t=None):
 line(im,pts,c,w);x,y=pts[-1];u,v=pts[-2];a=math.atan2(y-v,x-u)
 ImageDraw.Draw(im).polygon([(px(x),px(y)),(px(x-8*math.cos(a-.45)),px(y-8*math.sin(a-.45))),(px(x-8*math.cos(a+.45)),px(y-8*math.sin(a+.45)))],fill=c)
 if t is not None:
  lens=np.r_[0,np.cumsum([math.dist(a,b) for a,b in zip(pts,pts[1:])])];d=(t%1)*lens[-1]
  k=min(len(pts)-2,max(0,int(np.searchsorted(lens,d,side='right')-1)));f=(d-lens[k])/max(1e-8,lens[k+1]-lens[k])
  x=pts[k][0]*(1-f)+pts[k+1][0]*f;y=pts[k][1]*(1-f)+pts[k+1][1]*f
  dot(im,x,y,8,PANEL);dot(im,x,y,4,c)
def bez(a,b,c,d):
 return [tuple((1-u)**3*np.array(a)+3*(1-u)**2*u*np.array(b)+3*(1-u)*u*u*np.array(c)+u**3*np.array(d)) for u in np.linspace(0,1,40)]

def background():
 yy,xx=np.mgrid[0:px(H),0:px(W)];glow=np.exp(-((xx-px(930))**2/(px(650)**2)+(yy-px(70))**2/(px(490)**2)))
 a=np.zeros((*glow.shape,3),np.uint8)
 for k,(base,amp) in enumerate([(16,20),(11,9),(28,30)]):a[:,:,k]=base+amp*glow
 return Image.fromarray(a)
BASE=background()
def photo(path,size):
 k=(str(path),size)
 if k not in IC:IC[k]=ImageOps.fit(Image.open(path).convert('RGB'),(px(size),px(size)),Image.Resampling.LANCZOS)
 return IC[k]
def gallery(im,folder,cx,cy,width,t,size=100):
 paths=sorted((ASSETS/folder).glob('*.jpg'));clock=t*len(paths)/ACTIVE;idx=int(clock);f=clock-idx;spacing=size+16
 layer=Image.new('RGB',(px(width),px(size)),PANEL)
 for j in range(-2,4):
  x=width/2-size/2+(j-f)*spacing
  layer.paste(photo(paths[(idx+j)%len(paths)],size),(px(x),0))
 mask=Image.new('L',layer.size,0);ImageDraw.Draw(mask).rounded_rectangle((0,0,*layer.size),radius=px(8),fill=255)
 im.paste(layer,(px(cx-width/2),px(cy-size/2)),mask)

RESULTS=json.loads((ASSETS/'reported-results.json').read_text())
def gain(im,x,y,a,b):
 rect(im,(x-49,y-18,x+49,y+18),'#35213F',r=8)
 text(im,x,y,f'+{b-a:.1f} pp',21,PURPLE,True,'mm')
def scores(im,y,a,b,label=None):
 if label:text(im,803,y,label,18,MUTED,True,'lm')
 text(im,920,y,f'{a:.1f}',30,MUTED,True,'mm')
 arrow(im,[(954,y),(984,y)],PURPLE,2)
 text(im,1032,y,f'{b:.1f}',34,PURPLE,True,'mm');gain(im,1160,y,a,b)

def breadth(im,t):
 text(im,32,24,'BETA (Ours)',40,PURPLE,True)
 rect(im,(1000,23,1248,70),'#30203F',PURPLE,r=24)
 text(im,1124,47,'1 API call / image',24,WHITE,True,'mm')
 text(im,34,84,'Vision classifiers. Vision-language models. One interface.',27,WHITE)
 for i,(dataset,folder,family,model) in enumerate([
  ('ImageNet-C','imagenetc','VISION CLASSIFIER','ViT-B/16'),
  ('EuroSAT','eurosat','VISION–LANGUAGE MODEL','CLIP ViT-B/16'),
  ('Derm7pt','derm7pt','VISION–LANGUAGE MODELS','CLIP / BiomedCLIP')]):
  y=140+i*184;rect(im,(24,y,1256,y+168),PANEL,BORDER,14)
  text(im,44,y+13,dataset,23,WHITE,True)
  gallery(im,folder,139,y+102,190,t,100)
  text(im,380,y+20,family,18,TEAL,True,'mm')
  rect(im,(258,y+45,510,y+132),'#150F22',BORDER,10)
  text(im,384,y+73,'Frozen API',23,WHITE,True,'mm')
  text(im,384,y+109,model,22,MUTED,True,'mm')
  arrow(im,[(239,y+99),(253,y+99)],TEAL,2.2,t/2.8+i*.17)
  arrow(im,[(514,y+89),(554,y+89)],TEAL,2.2,t/2.8+i*.17+.15)
  rect(im,(562,y+45,754,y+132),'#362047',PURPLE,11,1.6)
  text(im,658,y+74,'BETA (Ours)',25,PURPLE,True,'mm')
  text(im,658,y+109,'Local steering',21,WHITE,False,'mm')
  arrow(im,bez((580,y+134),(520,y+163),(192,y+163),(139,y+154)),PURPLE,1.5,t/3.4+i*.17)
  arrow(im,[(758,y+89),(785,y+89)],PURPLE,2.2,t/2.8+i*.17+.4)
  if i<2:
   text(im,920,y+40,'Source',18,MUTED,True,'mm');text(im,1032,y+40,'BETA',18,PURPLE,True,'mm')
   a,b=RESULTS[dataset]['source'],RESULTS[dataset]['beta'];scores(im,y+87,a,b)
   line(im,[(830,y+136),(1205,y+136)],BORDER,5)
   line(im,[(830,y+136),(830+375*b/70,y+136)],PURPLE,5)
   dot(im,830+375*b/70*((t/4+i*.2)%1),y+136,3,WHITE)
  else:
   text(im,920,y+29,'Source',17,MUTED,True,'mm');text(im,1032,y+29,'BETA',17,PURPLE,True,'mm')
   scores(im,y+75,55.9,58.6,'CLIP');scores(im,y+125,60.9,62.1,'Biomed')
 text(im,1237,709,'Reported accuracy (%)',18,MUTED,False,'rb')

CHART=json.loads((ASSETS/'paper-chart-data.json').read_text())
def stability(im,t):
 q=clamp(t/ACTIVE)
 text(im,32,24,'Adapt without collapse',40,WHITE,True)
 text(im,32,82,'ImageNet-C Contrast · ViT-B/16',26,MUTED)
 rect(im,(1005,25,1248,67),'#30203F',PURPLE,r=22);text(im,1126,47,'BETA (Ours)',26,PURPLE,True,'mm')
 rect(im,(24,137,315,680),PANEL,BORDER,14)
 text(im,169,164,'Continuous inputs',25,WHITE,True,'mm')
 gallery(im,'contrast',169,301,252,t,225)
 text(im,169,441,'Local adaptation',24,MUTED,True,'mm')
 for i,label in enumerate(['Consistency','Reliable inputs']):
  y=495+i*77;rect(im,(44,y,295,y+55),'#352044',PURPLE,9)
  text(im,169,y+28,label,24,WHITE,True,'mm')
 arrow(im,[(169,421),(169,466)],PURPLE,2,t/3)
 arrow(im,[(169,553),(169,570)],PURPLE,2,t/3+.2)
 arrow(im,bez((295,600),(319,600),(319,226),(296,226)),PURPLE,1.8,t/3.5)
 l,r,top,bot=370,1170,190,604
 xm=lambda x:l+(x-1)/787*(r-l);ym=lambda y:bot-y/70*(bot-top)
 for val in [0,35,70]:
  y=ym(val);line(im,[(l,y),(r,y)],BORDER,1)
  if val!=35:text(im,l-15,y,str(val),20,MUTED,False,'rm')
 text(im,l,147,'Accuracy (%) ↑',24,WHITE,True)
 specs=[('Without consistency',RED),('Without filtering',GOLD),('BETA (Ours)',PURPLE)]
 for x,(n,c) in zip([405,717,1013],specs):
  line(im,[(x-33,119),(x-11,119)],c,3);text(im,x,119,n,21,c,True,'lm')
 xval=1+787*q;ends=[]
 for s,(name,c) in zip(CHART['stability'],specs):
  a=np.asarray(s['points']);xs,ix=np.unique(a[:,0],return_index=True);ys=a[ix,1]
  n=np.searchsorted(xs,xval,side='right');xx=xs[:n].tolist();yy=ys[:n].tolist()
  val=float(np.interp(xval,xs,ys));xx.append(xval);yy.append(val)
  line(im,[(xm(x),ym(y)) for x,y in zip(xx,yy)],c,4 if name=='BETA (Ours)' else 2.3)
  dot(im,xm(xval),ym(val),6,BG);dot(im,xm(xval),ym(val),4,c);ends.append((ym(val),val,c))
 line(im,[(xm(xval),top),(xm(xval),bot)],'#675176',1)
 ends.sort();last=top-8
 for y,v,c in ends:
  yy=max(top+15,min(bot-12,y),last+31);last=yy
  text(im,min(xm(xval)+12,1191),yy,f'{v:.1f}',25,c,True,'lm')
 text(im,l,638,'1',20,MUTED,False,'lm');text(im,r,638,'788',20,MUTED,False,'rm')
 text(im,770,650,'Online adaptation steps',24,WHITE,False,'mm')

SPECS={'breadth':('beta-model-families.mp4',breadth),'stability':('beta-stability.mp4',stability)}
def render(key,t):
 im=BASE.copy();SPECS[key][1](im,min(t,ACTIVE));return im.resize((W,H),Image.Resampling.LANCZOS)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--only',choices=SPECS);p.add_argument('--preview',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for key in ([a.only] if a.only else SPECS):
  name,fn=SPECS[key]
  if a.preview:
   for t in [0,5,10,20]:render(key,t).save(a.output/f'{key}-{t}.png')
   continue
  f=a.output/name
  proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','slow','-threads','2','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(f)],stdin=subprocess.PIPE)
  for i in range(round((ACTIVE+HOLD)*FPS)):
   proc.stdin.write(render(key,i/FPS).tobytes())
   if i and i%(FPS*5)==0:print(key,i/FPS,flush=True)
  proc.stdin.close();assert proc.wait()==0
  render(key,12).save(a.output/f'{f.stem}-poster.png')
  metadata={'file':f.name,'width':W,'height':H,'fps':FPS,'duration_seconds':ACTIVE+HOLD,'active_seconds':ACTIVE,'hold_seconds':HOLD,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'images':'Representative real ImageNet-C, EuroSAT, Derm7pt inputs; not per-image predictions','data':'Reported final paper tables' if key=='breadth' else 'Original paper Figure batch_acc_contrast.pdf vector curves; actual sampled path, with display interpolation only'}
  f.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print('DONE',key,flush=True)
if __name__=='__main__':main()
