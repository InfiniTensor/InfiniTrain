#!/usr/bin/env node
const fs=require('fs');
const [graphPath, outputPath]=process.argv.slice(2);
try {
  const graph=JSON.parse(fs.readFileSync(graphPath,'utf8')); const issues=[]; const warnings=[];
  if(!Array.isArray(graph.nodes)) issues.push('graph.nodes is missing or not an array');
  if(!Array.isArray(graph.edges)) issues.push('graph.edges is missing or not an array');
  const ids=new Set(), seen=new Set();
  for(const [i,n] of (graph.nodes||[]).entries()) { if(!n.id) issues.push(`Node[${i}] missing id`); if(seen.has(n.id)) issues.push(`Duplicate node ID '${n.id}'`); seen.add(n.id); ids.add(n.id); if(!n.type||!n.name||!n.summary||!Array.isArray(n.tags)||!n.tags.length) issues.push(`Node[${i}] '${n.id}' missing required field`); }
  for(const [i,e] of (graph.edges||[]).entries()) { if(!ids.has(e.source)) issues.push(`Edge[${i}] source '${e.source}' not found`); if(!ids.has(e.target)) issues.push(`Edge[${i}] target '${e.target}' not found`); }
  const fileTypes=new Set(['file','config','document','service','pipeline','table','schema','resource','endpoint']); const fileNodes=(graph.nodes||[]).filter(n=>fileTypes.has(n.type)).map(n=>n.id); const assigned=new Set();
  if(!Array.isArray(graph.layers)) issues.push('graph.layers is missing or not an array');
  for(const l of (graph.layers||[])){for(const id of (l.nodeIds||[])){if(!ids.has(id)) issues.push(`Layer '${l.id}' refs missing node '${id}'`); if(assigned.has(id)) issues.push(`Node '${id}' appears in multiple layers`); assigned.add(id);}}
  for(const id of fileNodes) if(!assigned.has(id)) issues.push(`File node '${id}' not in any layer`);
  if(!Array.isArray(graph.tour)) issues.push('graph.tour is missing or not an array');
  for(const [i,s] of (graph.tour||[]).entries()){if(!s.order||!s.title||!s.description||!Array.isArray(s.nodeIds)) issues.push(`Tour step[${i}] missing required field`); for(const id of (s.nodeIds||[])) if(!ids.has(id)) issues.push(`Tour step[${i}] refs missing node '${id}'`);}
  const withEdges=new Set((graph.edges||[]).flatMap(e=>[e.source,e.target])); for(const n of graph.nodes||[]) if(!withEdges.has(n.id)) warnings.push(`Node '${n.id}' has no edges (orphan)`);
  const stats={totalNodes:(graph.nodes||[]).length,totalEdges:(graph.edges||[]).length,totalLayers:(graph.layers||[]).length,tourSteps:(graph.tour||[]).length,nodeTypes:(graph.nodes||[]).reduce((a,n)=>(a[n.type]=(a[n.type]||0)+1,a),{}),edgeTypes:(graph.edges||[]).reduce((a,e)=>(a[e.type]=(a[e.type]||0)+1,a),{})};
  fs.writeFileSync(outputPath,JSON.stringify({issues,warnings,stats},null,2)); process.exit(0);
} catch(err){process.stderr.write(err.message+'\n');process.exit(1);}
