"use strict";
// 原生前端：没有外部 CDN；所有数据来自本机 API，字符串使用 textContent 展示。
const $ = id => document.getElementById(id);
const app = {state:null, song:null, songOffset:0, audioUrl:null, result:null, busy:false, polling:false, tab:"monitor", songRequest:0, catalogRequest:0};
const colors = ["#247cba", "#ce7b32"];
const format = value => value == null ? "—" : Math.abs(value) < 0.001 && value !== 0 ? Number(value).toExponential(2) : Number(value).toLocaleString(undefined,{maximumFractionDigits:5});
const tell = message => { $("error").textContent=message; $("error").hidden=!message; };
const headers = () => $("token").value ? {Authorization:`Bearer ${$("token").value}`} : {};
async function request(path, options={}) {
  const response=await fetch(path,{...options, headers:{...headers(),...(options.headers||{})},cache:"no-store"});
  if(!response.ok){let detail;try{detail=(await response.json()).detail;}catch{detail=response.statusText;}throw new Error(typeof detail==="string"?detail:JSON.stringify(detail));}
  return response;
}
async function json(path, body) { return (await request(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{})).json(); }
function setupCanvas(id) {
  const canvas=$(id), w=canvas.clientWidth, h=canvas.clientHeight, dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);return {ctx,w,h};
}
function chart(id, series, logarithmic=false) {
  const {ctx,w,h}=setupCanvas(id); if(!w)return; const pad={l:64,r:15,t:17,b:29};
  const convert=y=>logarithmic?(y>0?Math.log10(y):null):y;
  const valid=series.flatMap(s=>s).filter(p=>p[1]!=null&&Number.isFinite(p[1])&&convert(p[1])!=null);
  ctx.font="11px Segoe UI, sans-serif";ctx.fillStyle="#809291";
  if(!valid.length){ctx.textAlign="center";ctx.fillText("等待对应训练记录",w/2,h/2);return;}
  let xmin=Math.min(...valid.map(p=>p[0])),xmax=Math.max(...valid.map(p=>p[0]));
  let ymin=Math.min(...valid.map(p=>convert(p[1]))),ymax=Math.max(...valid.map(p=>convert(p[1])));
  if(xmin===xmax){xmin-=.5;xmax+=.5;}const margin=Math.max((ymax-ymin)*.12,logarithmic?.1:Math.abs(ymax)*.02,.0001);ymin-=margin;ymax+=margin;
  const x=v=>pad.l+(w-pad.l-pad.r)*(v-xmin)/(xmax-xmin),y=v=>h-pad.b-(h-pad.t-pad.b)*(v-ymin)/(ymax-ymin);
  for(let i=0;i<4;i++){const v=ymin+(ymax-ymin)*i/3,yy=y(v);ctx.strokeStyle="#e6edeb";ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.textAlign="right";ctx.fillStyle="#78908e";ctx.fillText(logarithmic?Math.pow(10,v).toExponential(1):format(v),pad.l-8,yy+4);}
  ctx.textAlign="center";[xmin,(xmin+xmax)/2,xmax].forEach(v=>ctx.fillText(Number.isInteger(v)?v:v.toFixed(1),x(v),h-7));
  series.forEach((points,index)=>{ctx.strokeStyle=colors[index];ctx.fillStyle=colors[index];ctx.lineWidth=2;ctx.beginPath();let active=false;points.forEach(p=>{const v=p[1]==null?null:convert(p[1]);if(v==null||!Number.isFinite(v)){active=false;return;}if(active)ctx.lineTo(x(p[0]),y(v));else ctx.moveTo(x(p[0]),y(v));active=true;});ctx.stroke();if(points.length<30)points.forEach(p=>{const v=p[1]==null?null:convert(p[1]);if(v!=null&&Number.isFinite(v)){ctx.beginPath();ctx.arc(x(p[0]),y(v),2.5,0,Math.PI*2);ctx.fill();}});});
}
function drawCharts(){if(!app.state||app.tab!=="monitor")return;const h=app.state.history;
  chart("qaChart",[h.epoch.map(e=>[e.epoch+1,e.train.qa_loss]),h.epoch.map(e=>[e.epoch+1,e.val.qa_loss])]);
  chart("lossChart",[h.train_step.map(e=>[e.step,e.total_loss]),h.train_step.map(e=>[e.step,e.retrieval_loss])]);
  chart("gradChart",["cnn","transformer"].map(name=>h.module_diagnostics.map(e=>[e.step,e.modules[name]?.grad_l2_before_clip])),true);
  chart("updateChart",["cnn","transformer"].map(name=>h.module_diagnostics.map(e=>[e.step,e.modules[name]?.relative_update])),true);
}
function renderState(data){app.state=data;const p=data.progress, stages={starting:"准备中",training:"训练中",validating:"验证中",saving:"保存中",completed:"训练完成",stopped:"按计划停止",interrupted:"已中断",failed:"训练失败",unknown:"暂无状态记录"};
  $("runName").textContent=`实验 / ${data.run_name}`;$("phase").textContent=stages[p.phase]||p.phase;
  $("heartbeat").textContent=p.updated_at?`记录更新于 ${new Date(p.updated_at*1000).toLocaleTimeString()}`:"旧日志无进程状态记录";
  $("epoch").textContent=p.epoch!=null?`${p.epoch} / ${p.epochs}`:"—";$("batch").textContent=p.batches?`本轮微批 ${p.batch||0} / ${p.batches}`:"等待训练批次";
  $("step").textContent=p.step??"—";$("planned").textContent=p.total_updates?`计划 ${p.total_updates} 次更新`:"等待训练计划";
  const last=data.history.epoch.at(-1);$("valLoss").textContent=format(last?.val.qa_loss);$("valEpoch").textContent=last?`来自第 ${last.epoch+1} 轮`:"等待完整验证";$("tinyBadge").hidden=!p.tiny_test_only;
  const diag=data.history.module_diagnostics.at(-1);if(diag){$("diagStep").textContent=`更新 ${diag.step} · ${diag.update_applied?"优化器已执行":"优化器跳过更新"}`;$("diagnostics").replaceChildren();for(const [name,m] of Object.entries(diag.modules)){const tr=document.createElement("tr");const status=m.trainable_parameters===0?"已冻结":(!m.grad_finite||!m.parameter_finite?"数值异常":m.grad_l2_before_clip===0?"梯度为 0":m.update_l2===0?"参数未改变":"有梯度与更新");const values=[name==="cnn"?"CNN":"Transformer",format(m.grad_l2_before_clip),format(m.grad_l2_after_clip),format(m.update_l2),format(m.relative_update),m.gradient_parameter_fraction==null?"—":`${(m.gradient_parameter_fraction*100).toFixed(1)}%`,status];values.forEach((v,i)=>{const td=document.createElement("td");td.textContent=v;if(i===6)td.className=status==="有梯度与更新"?"good":"bad";tr.append(td);});$("diagnostics").append(tr);}}
  const selected=$("checkpoint").value, names=data.checkpoints.map(c=>c.name);if(JSON.stringify(names)!==$("checkpoint").dataset.names){$("checkpoint").replaceChildren();data.checkpoints.forEach(c=>{const option=new Option(`${c.name}${c.tiny_test_only?" · tiny 自检":""}`,c.name);$("checkpoint").add(option);});if(!names.length)$("checkpoint").add(new Option("尚无完整检查点",""));$("checkpoint").value=names.includes(selected)?selected:names.at(-1)||"";$("checkpoint").dataset.names=JSON.stringify(names);}
  if(!app.busy)$("modelStatus").textContent=data.loaded_checkpoint?`已加载 ${data.loaded_checkpoint} · ${data.inference_device}`:`尚未加载 · 推理设备 ${data.inference_device}`;
  $("inspectButton").disabled=app.busy||!app.song||!data.loaded_checkpoint;$("loadModel").disabled=app.busy||!names.length;
  $("historyNote").textContent=`每类曲线保留最近 ${data.history.history_limit_per_type} 条记录；损坏日志行 ${data.history.invalid_lines}。状态是最后写入记录，强制结束进程后可能不再更新。${p.error?" 训练错误："+p.error:""}`;
  drawCharts();
}
async function poll(){if(app.polling)return;app.polling=true;try{renderState(await json("/api/state"));$("connectionDot").className="dot ok";$("connectionText").textContent="日志已连接";}catch(e){$("connectionDot").className="dot fail";$("connectionText").textContent="连接失败";tell(e.message);}finally{app.polling=false;}}
async function loadSongs(append=false){const generation=++app.catalogRequest;try{if(!append){app.songOffset=0;$("songList").replaceChildren();}const data=await json(`/api/songs?query=${encodeURIComponent($("songSearch").value)}&offset=${app.songOffset}`);if(generation!==app.catalogRequest)return;$("songCount").textContent=`匹配 ${data.total} 首音乐`;$("splitCheck").textContent=data.split_check;for(const song of data.songs){const button=document.createElement("button");button.className="song-item";button.dataset.id=song.song_id;button.disabled=app.busy;const name=document.createElement("span"),info=document.createElement("small");name.textContent=song.song_id;info.textContent=`${song.questions} 条参考问答 · ${song.audio}`;button.append(name,info);button.onclick=()=>selectSong(song.song_id);$("songList").append(button);}app.songOffset+=data.songs.length;$("moreSongs").hidden=app.songOffset>=data.total;}catch(e){if(generation===app.catalogRequest)tell(e.message);}}
function clearResult(){app.result=null;$("featureSection").hidden=true;$("conditionBadge").hidden=true;$("answer").textContent="选择问题并点击生成，查看当前歌曲的模型回答。";$("answer").classList.add("empty");$("resultMeta").textContent="";$("downloadJson").disabled=true;$("downloadFeatures").disabled=true;}
async function selectSong(id){const requestId=++app.songRequest;try{tell("");const data=await json(`/api/songs/${encodeURIComponent(id)}`);const blob=await(await request(`/api/songs/${encodeURIComponent(id)}/audio`)).blob();if(requestId!==app.songRequest)return;
  clearResult();if(app.audioUrl)URL.revokeObjectURL(app.audioUrl);app.audioUrl=URL.createObjectURL(blob);$("audio").src=app.audioUrl;app.song=id;$("songTitle").textContent=id;document.querySelectorAll(".song-item").forEach(b=>b.classList.toggle("selected",b.dataset.id===id));$("references").replaceChildren();data.references.forEach(ref=>{const card=document.createElement("article"),q=document.createElement("h4"),a=document.createElement("p"),button=document.createElement("button");q.textContent=ref.question;a.textContent=ref.answer;button.textContent="使用这道问题";button.className="quiet";button.onclick=()=>{$("question").value=ref.question;};card.append(q,a,button);$("references").append(card);});$("inspectButton").disabled=app.busy||!app.state?.loaded_checkpoint;
  }catch(e){if(requestId===app.songRequest)tell(e.message);}}
function heatmap(id, data){const {ctx,w,h}=setupCanvas(id);if(!w)return;const values=data.heatmap.values,rows=values.length,cols=values[0].length;const max=Math.max(...values.flat().map(Math.abs),1e-8),left=40,top=10,cw=(w-left-12)/cols,ch=(h-top-28)/rows;
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const t=values[r][c]/max;ctx.fillStyle=t>=0?`rgb(${Math.round(244-211*t)},${Math.round(248-121*t)},${Math.round(247-101*t)})`:`rgb(${Math.round(244+7*(-t))},${Math.round(248-124*(-t))},${Math.round(247-170*(-t))})`;ctx.fillRect(left+c*cw,top+r*ch,cw+.5,ch+.5);}
  ctx.font="10px Segoe UI, sans-serif";ctx.fillStyle="#6b827f";ctx.textAlign="right";ctx.fillText("0",left-8,top+8);ctx.fillText(String(data.heatmap.row_indices.at(-1)),left-8,h-30);ctx.textAlign="left";ctx.fillText("维度 0",left,h-8);ctx.textAlign="right";ctx.fillText(`维度 ${data.heatmap.column_indices.at(-1)}  |  ±${format(max)}`,w-12,h-8);
}
function showResult(result){app.result=result;$("answer").classList.remove("empty");$("answer").textContent=result.answer||"模型本次返回空文本。";$("resultMeta").textContent=`${result.checkpoint}${result.tiny_test_only?" · tiny 随机模型，输出不代表音乐能力":""}`;$("conditionBadge").hidden=false;$("conditionBadge").textContent=result.condition==="silence"?"静音对照":"原始音频";$("featureSection").hidden=false;
  for(const name of ["cnn","transformer"]){const data=result[name];$(name+"Shape").textContent=data.shape.join(" × ");$(name+"Stats").textContent=`均值 ${format(data.mean)} · 标准差 ${format(data.std)} · L2 ${format(data.l2)} · 范围 ${format(data.min)} ~ ${format(data.max)}`;$(name+"Preview").textContent=JSON.stringify(data.preview,null,2);heatmap(name+"Heatmap",data);}
  $("maskNote").textContent=`有效片段：${result.segment_mask.map(v=>v?"有效":"补零").join(" / ")}。CNN 卷积原始输出形状：${result.cnn_raw_shape.join(" × ")}。本次结果歌曲：${result.song_id}；问题：${result.question}`;$("downloadJson").disabled=false;$("downloadFeatures").disabled=false;
}
function busy(value){app.busy=value;$("loadModel").disabled=value;$("inspectButton").disabled=value||!app.song||!app.state?.loaded_checkpoint;for(const id of ["question","condition","tokens","checkpoint"])$(id).disabled=value;document.querySelectorAll(".song-item").forEach(button=>button.disabled=value);}
function saveBlob(blob, filename){const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$("loadModel").onclick=async()=>{busy(true);tell("");clearResult();$("modelStatus").textContent="正在加载模型，请稍候…";try{const data=await json("/api/load",{checkpoint:$("checkpoint").value});$("modelStatus").textContent=`已加载 ${data.checkpoint} · ${data.device}`;}catch(e){tell(e.message);}finally{busy(false);await poll();}};
$("inspectButton").onclick=async()=>{if(!app.song)return;busy(true);tell("");$("inspectButton").textContent="正在读取音乐并生成…";try{showResult(await json("/api/inspect",{song_id:app.song,question:$("question").value,max_new_tokens:Number($("tokens").value),condition:$("condition").value}));}catch(e){tell(e.message);}finally{busy(false);$("inspectButton").textContent="生成回答并检查";}};
$("downloadJson").onclick=()=>saveBlob(new Blob([JSON.stringify(app.result,null,2)],{type:"application/json"}),"inspection.json");
$("downloadFeatures").onclick=async()=>{try{saveBlob(await(await request(`/api/features/${app.result.download_id}`)).blob(),"audio_features.npz");}catch(e){tell(e.message);}};
document.querySelectorAll(".tab").forEach(button=>button.onclick=()=>{app.tab=button.dataset.tab;document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("active",b===button));$("monitorPanel").hidden=app.tab!=="monitor";$("inspectPanel").hidden=app.tab!=="inspect";if(app.tab==="inspect"){loadSongs();if(app.result)showResult(app.result);}else drawCharts();});
let searchTimer;$("songSearch").oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>loadSongs(),250);};$("moreSongs").onclick=()=>loadSongs(true);
$("refresh").onclick=()=>{tell("");poll();};$("connect").onclick=()=>{tell("");poll();if(app.tab==="inspect")loadSongs();};
window.addEventListener("resize",()=>{drawCharts();if(app.result&&app.tab==="inspect"){heatmap("cnnHeatmap",app.result.cnn);heatmap("transformerHeatmap",app.result.transformer);}});
poll();setInterval(poll,3000);
