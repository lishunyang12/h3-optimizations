#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the static public cross-SKU report from pinned case profiles."""
import argparse,csv,html,json,statistics,subprocess
from pathlib import Path
from sku_benchmark import ROOT, CATALOG, get_case, load, save


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--font',type=Path,required=True,help='NotoSansCJKsc-Regular.otf (SIL OFL)');args=p.parse_args()
 site=args.site;assets=site/'benchmarks/sku-reproduction';assets.mkdir(parents=True,exist_ok=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 rows=[]
 for name in load(CATALOG)['cases']:
  c=get_case(name);h=c['historical'];samples=h.get('http_samples_s',h.get('samples_s',[]))
  rows.append({'id':name,'family':c['family'],'gpu_count':c['topology']['num_gpus'],'gpu':'NVIDIA RTX PRO 5000 Blackwell',
   'width':c['media']['width'],'height':c['media']['height'],'frames':c['media']['frames'],'duration_seconds':c['media']['duration_seconds'],
   'steps':8 if c['family']=='omni5' else 4,'samples_s':samples,'median_s':statistics.median(samples),
   'n':len(samples),'historical':h,'request':c['request'],'protocol':c['protocol'],'reproduction_status':'packaged; fresh GPU rerun pending',
   'source_commit':commit})
 data={'schema':1,'publication_date':'2026-10-27','source_repository':'https://github.com/lishunyang12/h3-optimizations','source_commit':commit,
  'status':'Historical measurements; new launcher CPU/source/log checks passed; no fresh GPU measurements',
  'target_sku_status':'5500 exact hardware and results pending','cases':rows}
 save(assets/'results.json',data)
 with (assets/'timings.csv').open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['case','gpu_count','gpu','sample','historical_seconds','boundary'])
  for r in rows:
   for i,t in enumerate(r['samples_s'],1):w.writerow([r['id'],r['gpu_count'],r['gpu'],i,t,'historical HTTP; see case protocol'])
   for i,t in enumerate(r['historical'].get('ready_samples_s',[]),1):w.writerow([r['id'],r['gpu_count'],r['gpu'],i,t,'avatar application ready, includes checks'])
 save(assets/'avatar5-history.json',load(ROOT/'evidence/sku-avatar5-history.json'))
 save(assets/'omni5-history-inventory.json',load(ROOT/'evidence/sku-history-inventory.json'))
 save(assets/'validation.json',load(ROOT/'evidence/sku-package-validation.json'))
 body=[]
 for r in rows:
  body.append('<tr data-family="'+r['family']+'"><th scope="row"><code>'+r['id']+'</code></th><td>'+str(r['gpu_count'])+'</td><td>'+f"{r['width']} × {r['height']} / {r['frames']}f"+'</td><td>'+str(r['steps'])+'</td><td>'+f"{r['median_s']:.6f}"+'</td><td>'+str(r['n'])+'</td><td>待实测</td></tr>')
 payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
 doc='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5s / 15s 复现与测试数据 · H3</title>
<style>
@font-face{font-family:H3CJK;src:url(benchmarks/sku-reproduction/report-cjk.woff) format("woff");font-display:swap;unicode-range:U+80-10FFFF}:root{--bg:#f7f8f5;--ink:#1b2d2a;--muted:#546761;--green:#176b50;--line:#d6dfd8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 H3CJK,system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main,header,footer,nav{max-width:1180px;margin:auto;padding:24px 28px}nav{display:flex;gap:24px;flex-wrap:wrap;border-bottom:1px solid var(--line);font-size:14px}a{color:var(--green);text-underline-offset:4px}h1{font-size:clamp(32px,6vw,58px);font-weight:600;line-height:1.15;letter-spacing:-.04em;margin:16px 0 22px}h2{font-size:25px;font-weight:600;margin:0 0 18px}h3{font-size:18px;font-weight:600}p{max-width:940px}.eyebrow{color:var(--green);font:12px/1.6 ui-monospace,monospace;letter-spacing:.12em}.lead{font-size:19px;color:var(--muted);max-width:900px}.notice{border-left:4px solid #c49033;background:#fff7e5;padding:14px 20px}.links{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.button{display:inline-block;text-decoration:none;border:1px solid var(--green);padding:11px 17px;border-radius:5px;font-weight:600}.primary{background:var(--green);color:white}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:20px 0}.card{background:white;border:1px solid var(--line);padding:23px;border-radius:6px}.number{font:42px/1.2 ui-monospace,monospace;margin:15px 0 5px}.small{font-size:13px;color:var(--muted)}section{margin:20px 0 58px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:5px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:13px 15px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}thead{background:#eaf0e9}th{font-weight:500}code{font:13px/1.65 ui-monospace,H3CJK,monospace}pre{background:#172d28;color:#e8f4ec;padding:22px;border-radius:5px;overflow:auto;font-size:13px;line-height:1.8}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:16px 0}select{max-width:100%;padding:9px;border:1px solid #91aa9f;border-radius:3px;background:white;font:inherit}.chart{background:white;border:1px solid var(--line);padding:18px;border-radius:5px}.chart svg{width:100%;height:auto;display:block}.chart-label{font:12px ui-monospace,monospace;fill:#546761}a:focus-visible,select:focus-visible{outline:3px solid #be892a;outline-offset:4px}footer{border-top:1px solid var(--line);font-size:13px;overflow-wrap:anywhere}ul{padding-left:23px}li{margin:8px 0}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pill{font-size:12px;color:var(--green);padding:4px 8px;background:#e7f1e9;display:inline-block;border-radius:3px}@media(max-width:720px){main,header,nav,footer{padding:20px 16px}.cards,.columns{grid-template-columns:1fr}.number{font-size:36px}.lead{font-size:17px}.button{font-size:14px}section{margin-bottom:38px}}
</style></head><body>
<nav aria-label="主导航"><a href="index.html">← FastH3 消融报告</a><a href="avatar-quality.html">数字人画质基线</a><a href="five-second.html">5s 768p 原始报告</a><a href="#download">复现下载</a><a href="#data">测试数据</a></nav>
<header><div class="eyebrow">REPRODUCIBILITY · CROSS-SKU · 2026-10-27</div><h1>5 秒数字人，15 秒 FastH3。<br>同一配置，跨卡复现。</h1><p class="lead">固定模型、输入、精度、GPU 数和计时边界，保留 RTX PRO 5000 历史数据。将同一套脚本带到目标 SKU，生成可以核对的结果。</p>
<div class="notice"><strong>当前状态：历史实测 + 复现包已发布。</strong> 新启动器已完成 CPU、源码重建、历史日志和本地 HTTP / 音视频解码检查；尚无本次封装后的 GPU 复跑结果。5500 的具体型号和测试数字待实测，空白不代表失败。</div>
<div class="links" id="download"><a class="button primary" download href="benchmarks/sku-reproduction/h3-5s-15s-reproduction.zip">下载完整源码与复现 ZIP ↓</a><a class="button" href="__GUIDE__">环境与运行说明 ↗</a><a class="button" download href="benchmarks/sku-reproduction/results.json">测试数据 JSON ↓</a><a class="button" download href="benchmarks/sku-reproduction/timings.csv">逐次样本 CSV ↓</a></div><p class="small">ZIP 包含源码、补丁、参考图、固定请求和启动器，不含模型权重。<a href="benchmarks/sku-reproduction/downloads.json">下载 SHA-256 与字节数</a> · 源码版本 <a href="__COMMIT_URL__"><code>__SHORT_COMMIT__</code></a></p></header>
<main><section aria-labelledby="main-title"><h2 id="main-title">RTX PRO 5000 历史参考</h2><p>以下三组均使用 <strong>8 张 GPU</strong>。这是各自完整配置的请求耗时，不能当作单 GPU 结果，或单项量化的因果加速比。</p>
<div class="cards"><article class="card"><span class="pill">5s · 上线数字人路线</span><h3>Ref2VA Turbo · 4 步</h3><div class="number">__AVATAR_MEDIAN__ s</div><p class="small">13 个正式续段的 HTTP 中位数。960×544；124 帧窗口，重叠 5 帧，正式段播放 4.958 s。</p><p class="small">MXFP8 DiT · Dense Sage · NVFP4 视频 VAE。历史应用 ready 中位数 3.706 s，另含应用检查。</p></article>
<article class="card"><span class="pill">15s · 公开参考</span><h3>FastH3 · 4 步</h3><div class="number">13.304 s</div><p class="small">3 次正式请求中位数。1280×704，362 帧 / 15.083 s；TP1 / SP8。</p><p class="small">MXFP8 DiT · Sage/VSA · 完整 NVFP4 VAE · BF16 RDMA。</p></article>
<article class="card"><span class="pill">15s · 后续输出融合</span><h3>相同工作负载 + 融合</h3><div class="number">13.261 s</div><p class="small">另一轮的 3 次正式请求。该轮基线中位数为 13.334 s，新增输出融合。</p><p class="small">保留对应源码覆盖文件与融合调用审计；不声称这是全部历史中的最快值。</p></article></div>
<p class="small"><a href="avatar-quality.html">原数字人画质页</a>展示的是独立 10 秒 / 243 帧样本；页面中的 6.993 s 不能作为 5 秒时延。此处 5 秒复现来自保存的 16 次请求序列，排除前 3 次预热。<a href="benchmarks/sku-reproduction/avatar5-history.json">原始时延与媒体校验记录</a>。</p></section>
<section id="data"><h2>12 个固定配置与逐次数据</h2><p>9 月四步原生路线与 10 月八步 Omni 路线分别列示。所有数字均为历史 RTX PRO 5000 数据；严格跨 SKU 比较需两台机器重新运行同一脚本。</p>
<div class="controls"><label for="family">显示配置</label><select id="family"><option value="all">全部 12 个</option><option value="main">5s 数字人与 15s 主配置</option><option value="t2va5">5s / 768p 文生视频</option><option value="omni5">5s / 八步 FastH3-Omni</option></select></div>
<div class="table-wrap"><table><thead><tr><th>固定 case</th><th>GPU 数</th><th>输出</th><th>步数</th><th>历史中位数 / s</th><th>正式 n</th><th>5500</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<div class="controls"><label for="sample-case">查看原始样本</label><select id="sample-case"></select></div><div class="chart"><p id="chart-description" aria-live="polite"></p><svg id="chart" viewBox="0 0 960 220" role="img" aria-label="各次正式请求耗时，虚线为实际视频播放长度"></svg><p class="small">圆点按原始请求顺序排列；虚线是该 case 的实际播放长度。仅 3–13 次样本，不据此推断生产尾延迟。</p></div>
<p class="small">更早的 5s 原始 FastVideo / PR #5027、PR #5317 对照与同步回归在<a href="five-second.html">五秒性能原页</a>保留；它们有独立运行协议，未合并为本表的新测量。<a href="benchmarks/sku-reproduction/omni5-history-inventory.json">10 月其余已完成实验索引</a>。</p></section>
<section><h2>在 5000 和目标 SKU 上运行</h2><p>先按照<a href="__GUIDE__">完整说明</a>准备固定依赖、模型权重与 AdaLN 缓存，编辑机器路径。两台机器使用同一源码版本，保持相同 GPU 数、输入和缓存文件。</p>
<pre><code>git clone https://github.com/lishunyang12/h3-optimizations.git
cd h3-optimizations
git checkout __COMMIT__
python3 scripts/sku_benchmark.py prepare --variant avatar5 --output /opt/h3-runtime-avatar5
python3 scripts/sku_benchmark.py prepare --variant reference --output /opt/h3-runtime-15s
python3 scripts/sku_benchmark.py prepare --variant output-fusion --output /opt/h3-runtime-15s-output-fusion
cp configs/sku-assets.example.json /tmp/h3-assets.json
# 编辑路径、下载固定权重、构建缓存后：
python3 scripts/sku_benchmark.py suite --cases main \
  --assets /tmp/h3-assets.json --gpus 0,1,2,3,4,5,6,7 \
  --output /results/pro5000
# 目标机器使用同一命令，将输出目录改为 /results/pro5500。</code></pre>
<pre><code>python3 scripts/sku_benchmark.py compare \
  /results/pro5000/5s-avatar-ref2va-4step/result.json \
  /results/pro5500/5s-avatar-ref2va-4step/result.json \
  --output /results/avatar-comparison.json</code></pre>
<p class="small">两个 15 秒 case 同样使用 compare。4-GPU 的八步 Omni 配置须单独运行。GPU 架构、数量、显存或空闲条件不符合时，程序退出并给出原因。</p></section>
<section><h2>如何保证比较口径一致</h2><div class="columns"><div><h3>固定并记录</h3><ul><li>原始提示词、参考图、种子、帧数、步数及精度配置。</li><li>模型与缓存字节哈希、实际源码与二进制指纹、依赖版本。</li><li>GPU 型号、显存峰值、驱动、频率、功耗、CPU 与 PCIe / RDMA 拓扑。</li><li>预热单独保存；计时覆盖本地请求至完整 MP4 落盘。</li></ul></div><div><h3>检查与解释</h3><ul><li>四步运行检查真实 DiT 次数、量化 fallback、VAE 和 RDMA 证据。</li><li>正式计时外检查视频帧数与完整音视频解码；保留 MP4 和日志。</li><li>不同配置、权重、源码、依赖或受外部 GPU 作业干扰的结果拒绝配对。</li><li>CPU、互连、功耗和频率仍影响端到端时间；这是同工作负载的系统比较。</li></ul></div></div><p>复现数字人推理序列不包含聊天、RTMP 或直播播放延迟。量化配置属于有损路线；不同 GPU 的输出字节差异单独报告，不自动等价于画质优劣。</p><p><a href="benchmarks/sku-reproduction/validation.json">本次封装验证记录</a> · <a href="__SOURCE__">启动器源码</a> · <a href="__CASES__">全部固定配置</a> · <a href="post-dit.html">VAE 与输出优化历史</a></p></section></main>
<footer>历史测量来自保存的运行记录；此页没有填写推测或模拟的 GPU 结果。源码 <code>__COMMIT__</code> · 数据整理日期 2026-10-27。<a href="FONT-LICENSE.txt">Noto 字体许可</a></footer>
<script id="dataset" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.querySelector('#dataset').textContent), chooser=document.querySelector('#sample-case'), ns='http://www.w3.org/2000/svg';
for(const r of data.cases){const o=document.createElement('option');o.value=r.id;o.textContent=r.id;chooser.append(o)}
function el(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;document.querySelector('#chart').append(e);return e}
function draw(){const r=data.cases.find(x=>x.id===chooser.value), max=Math.max(...r.samples_s,r.duration_seconds)*1.13, x=v=>75+v/max*845;document.querySelector('#chart').replaceChildren();
 document.querySelector('#chart-description').textContent=r.id+' · '+r.gpu_count+' GPU · 中位数 '+r.median_s.toFixed(6)+' s · 正式 n='+r.n;
 el('line',{x1:75,y1:172,x2:920,y2:172,stroke:'#9aafa5'});for(let i=0;i<=5;i++){const v=max*i/5;el('line',{x1:x(v),x2:x(v),y1:45,y2:178,stroke:'#e1e8e2'});el('text',{x:x(v),y:202,'text-anchor':'middle',class:'chart-label'},v.toFixed(1)+'s')}
 el('line',{x1:x(r.duration_seconds),x2:x(r.duration_seconds),y1:35,y2:175,stroke:'#ba882d','stroke-dasharray':'6 5'});el('text',{x:x(r.duration_seconds),y:22,'text-anchor':'middle',class:'chart-label'},'播放 '+r.duration_seconds.toFixed(3)+'s');
 r.samples_s.forEach((v,i)=>{const dot=el('circle',{cx:x(v),cy:55+i*95/Math.max(1,r.n-1),r:5,fill:'#176b50'});const title=document.createElementNS(ns,'title');title.textContent='样本 '+(i+1)+': '+v.toFixed(6)+' s';dot.append(title)});
}
chooser.addEventListener('change',draw);draw();document.querySelector('#family').addEventListener('change',e=>{for(const tr of document.querySelectorAll('tbody tr')){const f=tr.dataset.family;tr.hidden=!(e.target.value==='all'||e.target.value===f||(e.target.value==='main'&&['avatar5','legacy15'].includes(f)))}});
</script></body></html>'''
 url='https://github.com/lishunyang12/h3-optimizations'
 for k,v in {'ROWS':''.join(body),'DATA':payload,'AVATAR_MEDIAN':f"{rows[0]['median_s']:.3f}",'COMMIT':commit,'SHORT_COMMIT':commit[:12],
  'COMMIT_URL':url+'/commit/'+commit,'GUIDE':url+'/blob/'+commit+'/docs/CROSS_SKU_REPRO.md',
  'SOURCE':url+'/blob/'+commit+'/scripts/sku_benchmark.py','CASES':url+'/blob/'+commit+'/configs/sku-cases.json'}.items():doc=doc.replace('__'+k+'__',v)
 from fontTools import subset
 from fontTools.ttLib import TTFont
 font=TTFont(str(args.font));sub=subset.Subsetter();sub.populate(text=doc);sub.subset(font)
 assert not (set(ord(c) for c in doc if ord(c)>127)-set(font.getBestCmap()))
 font.flavor='woff';font.save(assets/'report-cjk.woff')
 (site/'sku-reproduction.html').write_text(doc)
 index=site/'index.html'
 if index.exists() and 'href="sku-reproduction.html"' not in index.read_text():
  text=index.read_text();needle='<div class="nav-links">';assert needle in text
  index.write_text(text.replace(needle,needle+'<a href="sku-reproduction.html">5s / 15s 复现与数据</a>',1))
 print(json.dumps({'cases':len(rows),'site':str(site),'source_commit':commit}))

if __name__=='__main__':main()
