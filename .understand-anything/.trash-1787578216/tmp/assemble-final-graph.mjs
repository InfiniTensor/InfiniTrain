import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
const root='/home/fabu/桌面/InfiniTrain';
const inter=path.join(root,'.understand-anything','intermediate');
const base=JSON.parse(fs.readFileSync(path.join(inter,'assembled-graph.json'),'utf8'));
const scan=JSON.parse(fs.readFileSync(path.join(inter,'scan-result.json'),'utf8'));
const layers=JSON.parse(fs.readFileSync(path.join(inter,'layers.json'),'utf8'));
const tour=JSON.parse(fs.readFileSync(path.join(inter,'tour.json'),'utf8'));
const commit=execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
const graph={
  version:'1.0.0',
  project:{name:scan.name,languages:scan.languages,frameworks:scan.frameworks,description:scan.description,analyzedAt:new Date().toISOString(),gitCommitHash:commit},
  nodes:base.nodes,
  edges:base.edges,
  layers,
  tour
};
for (const n of graph.nodes) {
  if (!n.summary) n.summary='暂无摘要';
  if (!Array.isArray(n.tags)||!n.tags.length) n.tags=['untagged'];
}
const ids=new Set(graph.nodes.map(n=>n.id));
graph.edges=graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));
graph.layers=graph.layers.map((l,i)=>({id:l.id||`layer:layer-${i+1}`,name:l.name||`层 ${i+1}`,description:l.description||'项目架构层',nodeIds:(l.nodeIds||[]).filter(id=>ids.has(id))}));
graph.tour=graph.tour.map((s,i)=>({order:Number.isFinite(s.order)?s.order:i+1,title:s.title||`导览步骤 ${i+1}`,description:s.description||'查看相关代码与文档。',nodeIds:(s.nodeIds||[]).filter(id=>ids.has(id)),...(typeof s.languageLesson==='string'?{languageLesson:s.languageLesson}:{})})).sort((a,b)=>a.order-b.order);
fs.writeFileSync(path.join(inter,'assembled-graph.json'),JSON.stringify(graph,null,2)+'\n');
console.log(JSON.stringify({nodes:graph.nodes.length,edges:graph.edges.length,layers:graph.layers.length,tour:graph.tour.length,commit},null,2));
