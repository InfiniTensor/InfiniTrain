#!/usr/bin/env node
const fs = require('fs');
if (process.argv.length < 4) { console.error('usage: input output'); process.exit(1); }
let x;
try { x = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')); } catch (e) { console.error(e.message); process.exit(1); }
const ns = x.fileNodes || [], ids = new Set(ns.map(n => n.id)), by = new Map(ns.map(n => [n.id, n]));
const group = p => { const a = String(p || '').split('/'); return a.length > 1 ? a[0] : '<root>'; };
const dg = {}, tg = {};
for (const n of ns) { (dg[group(n.filePath)] ||= []).push(n.id); (tg[n.type] ||= []).push(n.id); }
const fi = {}, fo = {}, adj = {}, intra = {}, totals = {}, inter = {};
for (const n of ns) { fi[n.id] = 0; fo[n.id] = 0; adj[n.id] = []; }
for (const e of (x.importEdges || [])) {
  if (!ids.has(e.source) || !ids.has(e.target)) continue;
  fo[e.source]++; fi[e.target]++; adj[e.source].push(e.target);
  const a = group(by.get(e.source).filePath), b = group(by.get(e.target).filePath); totals[a] = (totals[a] || 0) + 1; totals[b] = (totals[b] || 0) + 1;
  if (a === b) intra[a] = (intra[a] || 0) + 1; const k = a + '\t' + b; inter[k] = (inter[k] || 0) + 1;
}
const cross = {};
for (const e of (x.allEdges || [])) { const a = by.get(e.source), b = by.get(e.target); if (!a || !b || a.type === b.type) continue; const k = a.type + '\t' + b.type + '\t' + e.type; cross[k] = (cross[k] || 0) + 1; }
const pattern = {routes:'api',api:'api',controllers:'api',handlers:'api',services:'service',core:'service',lib:'service',models:'data',data:'data',components:'ui',utils:'utility',common:'utility',tools:'utility',config:'config',tests:'test',test:'test',types:'types',cmd:'entry',internal:'service',docs:'documentation',deploy:'infrastructure',infra:'infrastructure',infrastructure:'infrastructure','.github':'ci-cd',k8s:'infrastructure',terraform:'infrastructure',sql:'data'};
const pm = Object.fromEntries(Object.keys(dg).map(g => [g, pattern[g] || 'other']));
const has = r => ns.some(n => r.test(n.filePath || ''));
const infraFiles = ns.filter(n => /(^|\/)Dockerfile|docker-compose|\.tf(?:vars)?$|(^|\/)(\.github|\.gitlab|\.circleci)(\/|$)|Jenkinsfile/.test(n.filePath || '')).map(n => n.filePath);
const result = {scriptCompleted:true,directoryGroups:dg,nodeTypeGroups:tg,importAdjacency:adj,fileFanIn:fi,fileFanOut:fo,interGroupImports:Object.entries(inter).map(([k,count])=>{const [from,to]=k.split('\t');return {from,to,count};}),intraGroupDensity:Object.fromEntries(Object.keys(dg).map(g=>[g,{internalEdges:intra[g]||0,totalEdges:totals[g]||0,density:(totals[g]?(intra[g]||0)/(totals[g]):0)}])),patternMatches:pm,crossCategoryEdges:Object.entries(cross).map(([k,count])=>{const [fromType,toType,edgeType]=k.split('\t');return {fromType,toType,edgeType,count};}),deploymentTopology:{hasDockerfile:has(/(^|\/)Dockerfile(?:\.|$)/),hasCompose:has(/docker-compose/),hasK8s:has(/(^|\/)(k8s|kubernetes|helm|charts)(\/|$)/),hasTerraform:has(/\.tf(?:vars)?$/),hasCI:has(/(^|\/)(\.github|\.gitlab|\.circleci)(\/|$)|Jenkinsfile/),infraFiles},fileStats:{totalFileNodes:ns.length,filesPerGroup:Object.fromEntries(Object.entries(dg).map(([k,v])=>[k,v.length])),nodeTypeCounts:Object.fromEntries(Object.entries(tg).map(([k,v])=>[k,v.length]))}};
try { fs.writeFileSync(process.argv[3], JSON.stringify(result, null, 2)); } catch (e) { console.error(e.message); process.exit(1); }
