import fs from 'node:fs';
import path from 'node:path';
const root='/home/fabu/桌面/InfiniTrain';
const inter=path.join(root,'.understand-anything','intermediate');
const g=JSON.parse(fs.readFileSync(path.join(inter,'assembled-graph.json'),'utf8'));
const fileTypes=new Set(['file','config','document','service','pipeline','table','schema','resource','endpoint']);
const files=g.nodes.filter(n=>fileTypes.has(n.type)&&n.filePath);
const layers=[
  ['layer:project-context','项目上下文与构建','README、CMake、CI、脚本和顶层配置，说明如何构建、测试、准备数据并启动训练。'],
  ['layer:core-runtime','核心运行时与张量','Tensor、Device、dtype、dispatcher、runtime guard、CCL 和公共基础设施。'],
  ['layer:autograd-nn','自动微分与神经网络模块','autograd Function、基础算子、Module/Transformer/LoRA 以及高层 functional API。'],
  ['layer:distributed-training','多维分布式训练','DDP、数据/张量/序列/流水线并行、process group、通信和分布式 optimizer。'],
  ['layer:backend-kernels','CPU/CUDA 后端 kernel','CPU 与 CUDA 算子、GEMM、dtype dispatch、gradient clipping 和 kernel 注册。'],
  ['layer:training-workflows','训练工作流与模型示例','MNIST、GPT-2、LLaMA、Mixtral 示例，以及 dataloader、optimizer、checkpoint 和 profiler。'],
  ['layer:tests-validation','测试与质量验证','单元测试、dtype/autograd/optimizer/parallel/transformer 测试和精度检查。'],
  ['layer:feature-guides','特性设计与实施指南','梯度裁剪、pipeline layout 等正在演进特性的设计、迁移、验证与交付文档。']
].map(([id,name,description])=>({id,name,description,nodeIds:[]}));
const choose=p=>{
  if (p==='README.md'||p==='CMakeLists.txt'||p.startsWith('.github/')||p.startsWith('cmake/')||p.startsWith('scripts/')||p==='LICENSE'||p==='.clang-format'||p==='.gitmodules'||p.startsWith('.understand-anything/')) return 0;
  if (p.startsWith('docs/')||p.startsWith('gradient_clipping_implementation/')||p.startsWith('pipeline_layout_implementation/')) return 7;
  if (p.startsWith('tests/')) return 6;
  if (p.startsWith('infini_train/src/kernels/')||p.startsWith('infini_train/include/common/cuda/')||p.startsWith('infini_train/src/core/runtime/cpu/')||p.startsWith('infini_train/src/core/runtime/cuda/')||p.startsWith('gradient_clipping_implementation/source/infini_train/src/kernels/')) return 4;
  if (p.startsWith('infini_train/include/nn/parallel/')||p.startsWith('infini_train/src/nn/parallel/')) return 3;
  if (p.startsWith('infini_train/include/autograd/')||p.startsWith('infini_train/src/autograd/')||p.startsWith('infini_train/include/nn/')||p.startsWith('infini_train/src/nn/')) return 2;
  if (p.startsWith('example/')||p.startsWith('tools/')||p.startsWith('infini_train/src/checkpoint/')||p.startsWith('infini_train/include/checkpoint/')||p.startsWith('infini_train/src/dataloader')||p.startsWith('infini_train/include/dataloader')||p.startsWith('infini_train/src/optimizer')||p.startsWith('infini_train/include/optimizer')||p.startsWith('infini_train/src/lr_scheduler')||p.startsWith('infini_train/include/lr_scheduler')||p.startsWith('infini_train/src/profiler')||p.startsWith('infini_train/include/profiler')||p.startsWith('infini_train/include/autocast')) return 5;
  return 1;
};
for (const n of files) layers[choose(n.filePath)].nodeIds.push(n.id);
for (const l of layers) l.nodeIds.sort();
fs.writeFileSync(path.join(inter,'layers.json'),JSON.stringify(layers,null,2)+'\n');
const ids=new Set(g.nodes.map(n=>n.id));
const issues=[]; const warnings=[]; const seen=new Set();
for(const [i,n] of g.nodes.entries()){if(!n.id||!n.type||!n.name||!n.summary||!Array.isArray(n.tags)||!n.tags.length) issues.push(`Node[${i}] missing required fields: ${n.id||'<no-id>'}`); if(seen.has(n.id)) issues.push(`Duplicate node ID: ${n.id}`); seen.add(n.id);}
for(const [i,e] of g.edges.entries()){if(!ids.has(e.source)||!ids.has(e.target)) issues.push(`Edge[${i}] dangling: ${e.source} -> ${e.target}`);}
const assigned=new Set(layers.flatMap(l=>l.nodeIds)); for(const n of files) if(!assigned.has(n.id)) issues.push(`File node not assigned: ${n.id}`); if(assigned.size!==files.length) warnings.push(`Layer assignment count ${assigned.size} differs from file node count ${files.length}`);
const withEdges=new Set(g.edges.flatMap(e=>[e.source,e.target])); for(const n of g.nodes) if(!withEdges.has(n.id)) warnings.push(`Orphan node: ${n.id}`);
fs.writeFileSync(path.join(inter,'assemble-review.json'),JSON.stringify({issues,warnings,stats:{totalNodes:g.nodes.length,totalEdges:g.edges.length,totalLayers:layers.length,fileNodes:files.length}},null,2)+'\n');
const by=(pred)=>files.filter(n=>pred(n.filePath)).map(n=>n.id);
const first=(pred, fallback)=>{const n=files.find(n=>pred(n.filePath)); return n?.id||fallback;};
const steps=[
 {order:1,title:'项目总览与构建入口',description:'从 README 和 CMake 构建配置开始，了解 InfiniTrain 的目标、依赖、编译选项和支持的训练模式。',nodeIds:[first(p=>p==='README.md','document:README.md'),first(p=>p==='CMakeLists.txt','file:CMakeLists.txt')]},
 {order:2,title:'核心张量与设备抽象',description:'阅读 Tensor、Device、datatype 和 dispatcher，理解算子如何在 CPU/CUDA 后端间选择实现。',nodeIds:by(p=>p==='infini_train/include/tensor.h'||p==='infini_train/src/tensor.cc'||p==='infini_train/include/device.h'||p==='infini_train/src/device.cc'||p==='infini_train/include/dispatcher.h')},
 {order:3,title:'自动微分引擎',description:'沿 autograd Function、grad mode 和基础反向算子，理解前向运算如何构建并执行反向图。',nodeIds:by(p=>p.startsWith('infini_train/include/autograd/')||p.startsWith('infini_train/src/autograd/')).slice(0,18)},
 {order:4,title:'神经网络模块与 Transformer',description:'从 Module、Linear、Normalization 进入 Transformer、attention、MLP 与 MoE 组件。',nodeIds:by(p=>p.startsWith('infini_train/include/nn/modules/')||p.startsWith('infini_train/src/nn/modules/')).slice(0,24)},
 {order:5,title:'分布式并行抽象',description:'理解 rank、process group、DDP、tensor/sequence parallel，以及 pipeline stage 和 schedule 如何组合。',nodeIds:by(p=>p.startsWith('infini_train/include/nn/parallel/')||p.startsWith('infini_train/src/nn/parallel/')).slice(0,28)},
 {order:6,title:'CPU 与 CUDA kernel',description:'查看 dispatcher 注册到 CPU/CUDA kernel 的路径，理解 GEMM、elementwise、reduction 等算子如何落地。',nodeIds:by(p=>p.startsWith('infini_train/src/kernels/')||p.startsWith('infini_train/include/common/cuda/')).slice(0,30)},
 {order:7,title:'训练示例与启动器',description:'以 MNIST、GPT-2、LLaMA 和 Mixtral 示例为入口，观察模型、数据、并行包装和训练循环的组合方式。',nodeIds:by(p=>p.startsWith('example/')||p.startsWith('tools/')).slice(0,30)},
 {order:8,title:'优化器、Checkpoint 与数据流',description:'继续阅读 optimizer、lr scheduler、dataloader 和 checkpoint，理解状态保存、恢复与参数更新。',nodeIds:by(p=>p.includes('/optimizer')||p.includes('/checkpoint/')||p.includes('/dataloader')||p.includes('/lr_scheduler')).slice(0,25)},
 {order:9,title:'测试与精度验证',description:'通过 tests 目录了解 tensor、autograd、dtype、并行、LoRA 和 Transformer 的验证边界。',nodeIds:by(p=>p.startsWith('tests/')).slice(0,35)},
 {order:10,title:'特性设计与演进路线',description:'最后阅读梯度裁剪和 pipeline layout 的设计、迁移与验证材料，理解当前工程如何从方案走向主干实现。',nodeIds:by(p=>p.startsWith('gradient_clipping_implementation/')||p.startsWith('pipeline_layout_implementation/')||p.startsWith('docs/')).slice(0,35)}
].map(s=>({...s,nodeIds:s.nodeIds.filter(id=>ids.has(id))}));
fs.writeFileSync(path.join(inter,'tour.json'),JSON.stringify(steps,null,2)+'\n');
console.log(JSON.stringify({layers:layers.map(l=>({id:l.id,name:l.name,count:l.nodeIds.length})),tourSteps:steps.length,issues:issues.length,warnings:warnings.length},null,2));
