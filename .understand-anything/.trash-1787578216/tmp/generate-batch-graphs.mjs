import fs from 'node:fs';
import path from 'node:path';

const root = '/home/fabu/桌面/InfiniTrain';
const inter = path.join(root, '.understand-anything', 'intermediate');
const tmp = path.join(root, '.understand-anything', 'tmp');
const minBatch = Number(process.env.MIN_BATCH ?? 15);
const maxBatch = Number(process.env.MAX_BATCH ?? 22);
const batches = JSON.parse(fs.readFileSync(path.join(inter, 'batches.json'), 'utf8')).batches
  .filter((b) => b.batchIndex >= minBatch && b.batchIndex <= maxBatch);

const special = new Map([
  ['docs/device_guard_design.md', '设计 DeviceGuard 运行时能力抽象、RAII 前端和 backend 静态注册机制。'],
  ['docs/dtype_registry_design.md', '设计低精度 dtype 抽象与 backend 注册流程，说明新增设备后端的扩展约定和失败模式。'],
  ['docs/gradient_clipping_background_and_source_guide.md', '从梯度、范数和 PyTorch 语义出发，导读 InfiniTrain 梯度裁剪调用链、分布式所有权与常见错误。'],
  ['docs/gradient_clipping_implementation_plan.md', '给出 ClipGradNorm 的 API、范数统计、dispatcher/kernel、并行语义、分阶段实现和验证矩阵。'],
  ['docs/hook_mechanism_design.md', '说明 module/function 的 forward/backward 四类 hook、handle 移除机制和调用时序。'],
  ['docs/lora_usage_guide.md', '介绍 LoRA 配置、注入、合并/卸载、权重保存加载、并行调用顺序以及 GPT2/LLaMA3 示例。'],
  ['docs/pipeline_layout_implementation_guide.md', '完整描述 Pipeline Layout 的数据结构、解析校验、模型/调度/checkpoint 接入、里程碑和数值验证。'],
  ['docs/pipeline_layout_public_validation_addendum.md', '记录 pipeline layout 公共源码复核结果、原计划需要的修正以及当前可交付范围。'],
  ['docs/precision_checker_guide.md', '说明 precision checker 的配置、输出格式、命令行用法、离线比较工具和 hook/counter 原理。'],
  ['docs/test_infrastructure_design.md', '设计设备参数化测试体系、CMake 宏、平台跳过规则以及新增设备后端的扩展步骤。'],
  ['docs/test_usage_guide.md', '提供测试构建、运行、新增 GTest 用例、条件执行宏和各测试模块注册指南。'],
  ['gradient_clipping_implementation/README.md', '概述梯度裁剪实现目录、设计文档、补丁、源码、测试和迁移顺序。'],
  ['gradient_clipping_implementation/design/00_background_and_source_guide.md', '梯度裁剪背景与源码导读，聚焦梯度所有权、分布式并行和现有调用链缺口。'],
  ['gradient_clipping_implementation/design/01_implementation_plan.md', '梯度裁剪实现计划，定义不变式、API、kernel、optimizer 接入、测试矩阵和完成标准。'],
  ['gradient_clipping_implementation/examples/integration_steps.md', '记录将梯度裁剪接入训练示例和 pipeline 训练循环的操作步骤。'],
  ['gradient_clipping_implementation/MIGRATION_CHECKLIST.md', '提供梯度裁剪迁移前的接口、并行语义、验证与文档检查清单。'],
  ['gradient_clipping_implementation/tests/clip_grad_norm_test_plan.md', '规划 ClipGradNorm 的单元、分布式和端到端测试覆盖。'],
  ['gradient_clipping_implementation/steps/00_migration_corrections.md', '列出迁移前需要修正的 device hint、统计/缩放分离、ZeRO、collective 和 PP/vPP 语义。'],
  ['gradient_clipping_implementation/steps/01_parameter_collection_code.md', '说明步骤一的逻辑参数集合代码及参数所有权入口。'],
  ['gradient_clipping_implementation/steps/02_core_api_code.md', '说明步骤二的梯度裁剪核心 API、参数校验和标量语义。'],
  ['gradient_clipping_implementation/steps/03_dispatcher_kernel_code.md', '说明步骤三的 dispatcher 与 CPU/CUDA kernel 接入方式。'],
  ['gradient_clipping_implementation/steps/04_metadata_code.md', '说明步骤四如何为 TP/共享参数增加副本角色和统计元数据。'],
  ['gradient_clipping_implementation/steps/05_zero_and_parallel_code.md', '说明步骤五的 DistributedOptimizer、ZeRO 子集和并行归约接入。'],
  ['gradient_clipping_implementation/steps/06_known_risks.md', '汇总 host scalar 读取、CUDA reduction、TP metadata 和 DistributedOptimizer 去重风险。'],
  ['gradient_clipping_implementation/steps/07_example_and_schedule_code.md', '说明步骤七如何更新 example 与 PP 训练调度代码。'],
  ['gradient_clipping_implementation/steps/08_verification_code.md', '说明步骤八的验证代码、oracle 和结果记录。'],
  ['gradient_clipping_implementation/steps/implementation_sequence.md', '按逻辑梯度、数学核心、dispatcher、参数副本和 optimizer 接入排列实现序列。'],
  ['pipeline_layout_implementation/chapters/03_background_knowledge/README.md', '介绍 pipeline layout 背景知识章节及其源码核对材料。'],
  ['pipeline_layout_implementation/chapters/03_background_knowledge/public_validation.md', '记录 pipeline layout 设计所依赖的公开源码和 API 复核结果。'],
  ['pipeline_layout_implementation/chapters/03_background_knowledge/source_check_commands.md', '汇总用于复核 pipeline、parallel 和 checkpoint 源码的命令。'],
  ['pipeline_layout_implementation/chapters/14_deliverables/report_template.md', '提供 pipeline layout 项目报告模板，覆盖实现范围、验证结果和风险。'],
  ['pipeline_layout_implementation/chapters/14_deliverables/standalone_test_log.md', '记录 standalone pipeline layout 测试命令、输入和结果。'],
  ['pipeline_layout_implementation/chapters/14_deliverables/test_log_template.md', '提供可复用的测试日志模板。'],
  ['pipeline_layout_implementation/chapters/14_deliverables/user_guide.md', '提供 standalone pipeline layout 的构建、CLI 和验证使用指南。'],
  ['pipeline_layout_implementation/README.md', '概述 standalone Pipeline Layout 实现、目录结构、构建方式和 CLI 示例。'],
  ['README.md', '项目总览与快速开始文档，涵盖构建、MNIST/GPT2/LLaMA3 示例、启动方式和 DDP/TP/PP 并行策略。'],
]);

function basename(p) { return p.split('/').at(-1); }
function ext(p) { const b = basename(p); const i = b.lastIndexOf('.'); return i > 0 ? b.slice(i).toLowerCase() : ''; }
function complexity(n) { return n < 50 ? 'simple' : n < 200 ? 'moderate' : 'complex'; }
function slug(s) { return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''); }
function topicTags(p) {
  const l = p.toLowerCase();
  const tags = [];
  if (l.includes('cuda') || /\.(cu|cuh)$/.test(l)) tags.push('cuda-kernel');
  if (l.includes('gradient_clipping') || l.includes('gradient_clip')) tags.push('gradient-clipping');
  if (l.includes('pipeline_layout') || l.includes('pipeline-layout')) tags.push('pipeline-parallelism');
  if (l.includes('test')) tags.push('test');
  if (l.includes('script') || l.endsWith('.sh') || l.endsWith('.bash') || l.endsWith('.py')) tags.push('automation');
  if (l.includes('cmake')) tags.push('build-system');
  if (l.includes('design') || l.includes('guide') || l.endsWith('.md')) tags.push('documentation');
  return tags;
}
function nodeType(f) {
  const p = f.path, l = p.toLowerCase(), e = ext(p);
  if (f.fileCategory === 'config' || basename(p) === '.clang-format' || basename(p) === '.gitmodules' || basename(p) === '.understandignore' || e === '.cmake' || basename(p) === 'cmakelists.txt') return 'config';
  if (f.fileCategory === 'docs') return 'document';
  if (f.fileCategory === 'infra') return l.includes('workflow') ? 'pipeline' : 'service';
  if (f.fileCategory === 'data') return e === '.sql' ? 'table' : 'schema';
  return 'file';
}
function fallbackSummary(f) {
  const p = f.path, l = p.toLowerCase(), b = basename(p);
  if (special.has(p)) return special.get(p);
  if (b === '.clang-format') return '统一 C/C++/CUDA 源码格式、缩进和 include 排序规则。';
  if (b === '.gitmodules') return '声明项目使用的 Git 子模块及其路径映射。';
  if (b === '.understandignore') return '控制 Understand-Anything 知识图谱扫描时需要排除的路径和文件。';
  if (b === 'config.json' && p.includes('.understand-anything')) return '保存知识图谱生成的输出语言等项目级配置。';
  if (b === 'CMakeLists.txt' || l.endsWith('.cmake')) return `CMake 构建配置，负责${l.includes('test') ? '注册测试目标和测试辅助宏' : '组织项目、库、CUDA/NCCL 依赖和编译选项'}。`;
  if (e(p) === '.cu' || e(p) === '.cuh') {
    if (l.includes('elementwise')) return 'CUDA elementwise kernel 集合，覆盖一元/二元广播、反向传播和多种标量运算。';
    if (l.includes('transform')) return 'CUDA 张量变换 kernel，实现转置、triangular mask、mask 和 repeat-interleave 的前向/反向。';
    if (l.includes('linear')) return 'CUDA 线性层实现，提供 bias copy、前向矩阵乘和输入/权重/bias 反向 kernel。';
    if (l.includes('reduction')) return '基于 CUB 的 CUDA reduction，实现 mean/sum/max/min 前向与反向。';
    if (l.includes('cross_entropy')) return 'CUDA cross entropy 前向与反向 kernel，支持不同输入和 target dtype。';
    if (l.includes('vocab_parallel')) return 'CUDA vocabulary-parallel cross entropy 的分片反向计算实现。';
    if (l.includes('layernorm')) return 'CUDA layer normalization 前向/反向 kernel，处理均值、方差和 affine 参数。';
    if (l.includes('softmax')) return 'CUDA softmax 前向/反向 kernel，按指定维度进行稳定归一化。';
    if (l.includes('matmul')) return 'CUDA matmul 前向以及两个输入梯度的反向实现。';
    if (l.includes('gemm')) return 'CUDA GEMM/SGEMV backend 适配层，封装设备 kernel 注册和矩阵乘调用。';
    if (l.includes('embedding')) return 'CUDA embedding lookup 前向和按索引累加权重梯度的反向实现。';
    if (l.includes('gather')) return 'CUDA gather 前向索引读取和反向梯度散射实现。';
    if (l.includes('scatter')) return 'CUDA scatter 前向写入和反向梯度收集实现。';
    if (l.includes('concat')) return 'CUDA concat 前向拼接与反向梯度切分实现。';
    if (l.includes('split')) return 'CUDA split 前向分片和反向合并实现。';
    if (l.includes('stack')) return 'CUDA stack 前向堆叠与反向梯度拆分实现。';
    if (l.includes('slice')) return 'CUDA slice 前向切片与反向梯度回填实现。';
    if (l.includes('outer')) return 'CUDA outer product 前向和输入梯度反向实现。';
    if (l.includes('topk')) return 'CUDA top-k 选择前向和索引梯度反向实现。';
    if (l.includes('fill')) return 'CUDA 张量填充 kernel，把 scalar 广播写入整块设备内存。';
    if (l.includes('accumulate_grad')) return 'CUDA 梯度累加与 Adam 状态更新 kernel。';
    if (l.includes('gradient_clip')) return 'CUDA 梯度范数统计和原地缩放 kernel，并通过 backend 注册表暴露。';
    if (l.includes('comm')) return 'CUDA 通信相关 tensor 操作实现，连接设备 kernel 与通信抽象。';
    if (l.includes('cast')) return 'CUDA dtype cast kernel，在设备上完成张量类型转换。';
    if (l.includes('no_op')) return 'CUDA NoOp 前向/反向实现，用于保持图结构但不改变数据。';
    if (l.includes('kernel_helper')) return 'CUDA 通用 device helper，提供 dtype cast、数学函数、广播原子加和 fastAtomicAdd。';
    if (l.includes('cub_compat')) return 'CUDA CUB 兼容头，统一不同 CUB 版本下的 include 和适配宏。';
  }
  if (l.endsWith('.py')) return `Python 工具脚本，负责${l.includes('precision') ? '精度结果的张量比较' : l.includes('format') ? '按 git 变更范围执行代码格式化' : '模型资源或训练辅助数据的准备'}。`;
  if (l.endsWith('.sh') || l.endsWith('.bash')) return `Shell 自动化脚本，负责${l.includes('profile') ? '构建、运行模型测试并收集 profile 日志' : '准备模型资源或执行 standalone 验证'}。`;
  if (l.endsWith('.patch')) return '梯度裁剪迁移补丁，展示需要应用到 optimizer、tensor metadata、通信 buffer 和 owner filtering 的代码变更。';
  if (p.includes('pipeline_layout_implementation/chapters')) return `Pipeline Layout 实现章节材料，覆盖${p.includes('unit_tests') ? '单元测试' : p.includes('numerical') ? '数值一致性比较' : p.includes('scheduler') ? '调度器接入' : p.includes('checkpoint') ? 'checkpoint 参数映射' : p.includes('model_integration') ? '模型布局适配' : p.includes('parser') ? '布局解析与校验' : '布局核心数据结构'}。`;
  return `${b} 项目文件，包含与 InfiniTrain 构建、运行或验证相关的配置和实现。`;
}
function fileTags(f, type) {
  const p=f.path,l=p.toLowerCase(), tags=new Set(topicTags(p));
  if (type === 'document') tags.add('documentation');
  if (type === 'config') tags.add('configuration');
  if (type === 'file') tags.add('implementation');
  if (l.endsWith('.md')) tags.add(l.includes('design') ? 'design' : 'guide');
  if (l.endsWith('.patch')) tags.add('migration');
  if (l.includes('pipeline_layout')) tags.add('layout');
  if (l.includes('cuda')) tags.add('device-backend');
  if (l.includes('scripts/assets')) tags.add('data-preparation');
  if (l.includes('run_models')) tags.add('profiling');
  if (l.includes('precision_check')) tags.add('numerical-validation');
  if (l.includes('cmake')) tags.add('cmake');
  while(tags.size<3) tags.add('infinitrain');
  return [...tags].slice(0,5);
}
function functionSummary(name, file) {
  const n=name.toLowerCase();
  if (n.includes('builddefault')) return '构建默认的连续 pipeline stage/chunk 布局并建立校验索引。';
  if (n.includes('buildcontiguous')) return '根据每个 stage 的层数构建连续分片布局。';
  if (n.includes('buildexplicit')) return '从显式 chunk 描述创建并校验 PipelineLayout。';
  if (n.includes('validate')) return '校验布局的 stage、chunk、特殊模块边界和当前传输约束。';
  if (n.includes('schedule')) return '生成或筛选与 pipeline layout 对应的 micro-batch 调度任务。';
  if (n.includes('loadplan') || n.includes('ownedgloballayers')) return '依据 stage 所有权生成 checkpoint 层加载计划。';
  if (n.includes('comparevectors')) return '按 precision 容差比较期望和实际向量并报告最坏误差。';
  if (n.includes('suggestcontiguous')) return '根据层成本建议连续 stage 划分并计算负载不均衡指标。';
  if (n.includes('parse')) return '解析命令行或布局字符串并将 token 转换为结构化参数。';
  if (n === 'main') return '命令行入口，解析参数并驱动构建、运行或验证流程。';
  if (file.endsWith('.py')) return `执行 ${name}，完成资源准备、格式化或精度比较流程中的一个步骤。`;
  if (file.endsWith('.sh') || file.endsWith('.bash')) return `执行 ${name}，封装构建、下载、测试或 profile 自动化操作。`;
  return `实现 ${name}，承担该模块中的核心计算或参数校验逻辑。`;
}
function classSummary(name) {
  const n=name.toLowerCase();
  if (n.includes('pipelinelayout')) return '表示层、stage、vPP 和特殊模块放置关系，并提供构建、查询与校验接口。';
  if (n.includes('schedule')) return '描述一个 layout-aware pipeline 调度任务及其 micro-batch/chunk 标记。';
  if (n.includes('modelplan')) return '表示 stage 内局部 chunk、层顺序和特殊模块挂载计划。';
  if (n.includes('layerload')) return '描述 checkpoint 层在 stage 上的加载决定和局部索引。';
  if (n.includes('comparison')) return '承载数值比较通过状态、误差统计和失败位置。';
  if (n.includes('costbalanced')) return '记录按计算成本平衡后的布局建议及 bubble/imbalance 指标。';
  if (n.includes('safetensorshard')) return '封装 SafeTensor 分片读取、索引查找和原始数组访问。';
  return `表示 ${name}，封装该模块的结构化状态或运行参数。`;
}
function isSignificant(item) {
  const span=(item.endLine||0)-(item.startLine||0)+1;
  return span>=10 || item._exported;
}

function addRelatedEdges(nodes, edges, files) {
  const byPath = new Map(files.map(f=>[f.path, f]));
  const add=(s,t,type='related',weight=0.5)=>{ if(s!==t && nodes.some(n=>n.id===s) && nodes.some(n=>n.id===t) && !edges.some(e=>e.source===s&&e.target===t&&e.type===type)) edges.push({source:s,target:t,type,direction:'forward',weight}); };
  for (const f of files) {
    const id = `${nodeType(f)}:${f.path}`;
    const l=f.path.toLowerCase();
    if (l.endsWith('readme.md')) {
      const siblings=files.filter(g=>g.path!==f.path && g.path.startsWith(f.path.slice(0,f.path.lastIndexOf('/')+1)) && /\.(cc|cpp|h|c|py|sh|bash|cu|cuh)$/.test(g.path));
      for (const g of siblings.slice(0,4)) add(id,`${nodeType(g)}:${g.path}`,'documents',0.5);
    }
    if (f.path.includes('CMakeLists.txt')) {
      const siblings=files.filter(g=>g.path!==f.path && g.path.startsWith(f.path.slice(0,f.path.lastIndexOf('/')+1)) && /\.(cc|cpp|h|c|py|sh|bash|cu|cuh)$/.test(g.path));
      for (const g of siblings.slice(0,5)) add(id,`${nodeType(g)}:${g.path}`,'configures',0.6);
    }
    if (l.endsWith('kernel_helper.cuh') || l.endsWith('gemm.cuh')) {
      const siblings=files.filter(g=>g.path!==f.path && g.path.includes('/kernels/cuda/') && g.path.endsWith('.cu'));
      for (const g of siblings.slice(0,8)) add(id,`${nodeType(g)}:${g.path}`,'depends_on',0.6);
    }
  }
}

for (const b of batches) {
  console.error(`start batch ${b.batchIndex}`);
  const ex = JSON.parse(fs.readFileSync(path.join(tmp, `ua-file-extract-results-${b.batchIndex}.json`), 'utf8'));
  const exportsByFile = new Map(ex.results.map(r=>[r.path,new Set((r.exports||[]).map(e=>e.name))]));
  const nodes=[]; const edges=[]; const fileNodeIds=new Map();
  for (const f of b.files) {
    const r=ex.results.find(x=>x.path===f.path) || {totalLines:f.sizeLines,nonEmptyLines:f.sizeLines,functions:[],classes:[],exports:[]};
    const type=nodeType(f); const id=`${type}:${f.path}`; fileNodeIds.set(f.path,id);
    const node={id,type,name:basename(f.path),filePath:f.path,summary:fallbackSummary(f),tags:fileTags(f,type),complexity:complexity(r.nonEmptyLines||f.sizeLines)};
    if ((f.language||'').toLowerCase().includes('cuda') || /\.(cu|cuh)$/.test(f.path)) node.languageNotes='CUDA 代码通过 backend 注册宏暴露前向/反向 kernel；本批次确定性提取器未展开 CUDA 函数级结构。';
    nodes.push(node);
  }
  console.error(`files done ${b.batchIndex}`);
  for (const r of ex.results) {
    const f=b.files.find(x=>x.path===r.path); if(!f) continue;
    const fileId=fileNodeIds.get(f.path); const exps=exportsByFile.get(f.path)||new Set();
    for (const fn of (r.functions||[])) { fn._exported=exps.has(fn.name); if(!isSignificant(fn)) continue;
      const id=`function:${f.path}:${fn.name}`; if(nodes.some(n=>n.id===id)) continue;
      nodes.push({id,type:'function',name:fn.name,filePath:f.path,lineRange:[fn.startLine,fn.endLine],summary:functionSummary(fn.name,f.path),tags:['function','implementation',...(f.path.includes('test')?['test']:[])].slice(0,5),complexity:complexity((fn.endLine||0)-(fn.startLine||0)+1)});
      edges.push({source:fileId,target:id,type:'contains',direction:'forward',weight:1.0});
      if(fn._exported) edges.push({source:fileId,target:id,type:'exports',direction:'forward',weight:0.8});
    }
    for (const cl of (r.classes||[])) { cl._exported=exps.has(cl.name); if(!isSignificant(cl)) continue;
      const id=`class:${f.path}:${cl.name}`; if(nodes.some(n=>n.id===id)) continue;
      nodes.push({id,type:'class',name:cl.name,filePath:f.path,lineRange:[cl.startLine,cl.endLine],summary:classSummary(cl.name),tags:['class','data-structure','implementation'].slice(0,5),complexity:complexity((cl.endLine||0)-(cl.startLine||0)+1)});
      edges.push({source:fileId,target:id,type:'contains',direction:'forward',weight:1.0});
      if(cl._exported) edges.push({source:fileId,target:id,type:'exports',direction:'forward',weight:0.8});
    }
  }
  console.error(`structural done ${b.batchIndex}`);
  for (const f of b.files) {
    const src=fileNodeIds.get(f.path); const imports=(b.batchImportData&&b.batchImportData[f.path])||[];
    for (const p of imports) edges.push({source:src,target:`file:${p}`,type:'imports',direction:'forward',weight:0.7});
  }
  console.error(`imports done ${b.batchIndex}`);
  addRelatedEdges(nodes,edges,b.files);
  console.error(`related done ${b.batchIndex}`);
  // Remove stale output parts for this batch before writing current fragments.
  for (const name of fs.readdirSync(inter)) if(new RegExp(`^batch-${b.batchIndex}(?:-part-\\d+)?\\.json$`).test(name)) fs.unlinkSync(path.join(inter,name));
  const nodeCount=nodes.length, edgeCount=edges.length;
  const parts=Math.max(1,Math.ceil(Math.max(nodeCount/60,edgeCount/120)));
  const sorted=[...b.files].sort((a,c)=>a.path.localeCompare(c.path));
  const chunkSize=Math.ceil(sorted.length/parts);
  for(let k=0;k<parts;k++) {
    const chunk=sorted.slice(k*chunkSize,(k+1)*chunkSize); const paths=new Set(chunk.map(f=>f.path));
    const partNodes=nodes.filter(n=>paths.has(n.filePath)); const partIds=new Set(partNodes.map(n=>n.id));
    const partEdges=edges.filter(e=>partIds.has(e.source));
    const out={nodes:partNodes,edges:partEdges};
    const file=parts===1?`batch-${b.batchIndex}.json`:`batch-${b.batchIndex}-part-${k+1}.json`;
    fs.writeFileSync(path.join(inter,file),JSON.stringify(out,null,2)+'\n');
    JSON.parse(fs.readFileSync(path.join(inter,file),'utf8'));
  }
  console.log(`batch ${b.batchIndex}: ${nodeCount} nodes, ${edgeCount} edges, ${parts} part(s)`);
}
