import fs from 'node:fs';
import path from 'node:path';

const root = '/home/fabu/桌面/InfiniTrain';
const inter = path.join(root, '.understand-anything', 'intermediate');
const scan = JSON.parse(fs.readFileSync(path.join(inter, 'scan-result.json'), 'utf8'));
const batches = JSON.parse(fs.readFileSync(path.join(inter, 'batches.json'), 'utf8'));
const imports = scan.importMap || {};
const fileByPath = new Map(scan.files.map(f => [f.path, f]));
const prefixes = { config: 'config', docs: 'document', infra: 'service', code: 'file', data: 'schema', script: 'file', markup: 'file' };
const idFor = f => {
  if (f.fileCategory === 'infra' && f.path.startsWith('.github/workflows/')) return `pipeline:${f.path}`;
  return `${prefixes[f.fileCategory] || 'file'}:${f.path}`;
};
const summaryFor = f => {
  const p = f.path;
  if (p === 'README.md') return '项目总览与快速开始文档，说明构建方式、训练示例、分布式启动参数和支持矩阵。';
  if (p.startsWith('docs/')) return `设计文档，记录 InfiniTrain 的${p.split('/').pop().replace(/\.[^.]+$/, '')}相关背景、接口或验证方法。`;
  if (p.startsWith('pipeline_layout_implementation/')) return '流水线布局实现与验证材料，覆盖布局解析、模型集成、调度、checkpoint 映射和测试。';
  if (p.startsWith('gradient_clipping_implementation/')) return '梯度裁剪功能的设计、迁移步骤、kernel 实现和测试材料。';
  if (p.startsWith('.github/workflows/')) return 'GitHub Actions 持续集成流程，用于格式检查和自动化验证。';
  if (p === 'CMakeLists.txt' || p.endsWith('.cmake')) return 'CMake 构建配置，定义 InfiniTrain 库、CPU/CUDA kernel、示例程序和测试目标。';
  if (p.endsWith('.cu') || p.endsWith('.cuh')) return 'CUDA 后端 kernel 或辅助头文件，为张量算子提供 GPU 执行实现。';
  if (p.endsWith('.cc') || p.endsWith('.cpp') || p.endsWith('.h') || p.endsWith('.hpp')) return 'C++ 源码或头文件，承担 InfiniTrain 的运行时、算子、并行训练或工具实现。';
  if (p.endsWith('.py') || p.endsWith('.sh') || p.endsWith('.bash')) return '项目脚本，用于资源准备、格式化、性能比较或训练启动。';
  return '项目辅助文件，参与构建、文档、测试或训练流程。';
};
const tagsFor = f => {
  const p = f.path;
  const tags = [];
  if (p === 'README.md') tags.push('documentation', 'entry-point', 'overview');
  else if (p.startsWith('docs/') || p.endsWith('.md')) tags.push('documentation', 'design', 'reference');
  else if (p.startsWith('pipeline_layout_implementation/')) tags.push('pipeline-parallel', 'implementation', 'validation');
  else if (p.startsWith('gradient_clipping_implementation/')) tags.push('gradient-clipping', 'implementation', 'testing');
  else if (p.startsWith('.github/')) tags.push('ci-cd', 'automation', 'validation');
  else if (p.endsWith('.cu') || p.endsWith('.cuh')) tags.push('cuda', 'kernel', 'gpu-compute');
  else if (p.endsWith('.cmake') || p === 'CMakeLists.txt') tags.push('configuration', 'build-system', 'cmake');
  else if (p.endsWith('.py') || p.endsWith('.sh') || p.endsWith('.bash')) tags.push('script', 'tooling', 'automation');
  else tags.push('c++', 'framework-core', 'implementation');
  return tags.slice(0, 5);
};
const typeFor = f => f.path.startsWith('.github/workflows/') ? 'pipeline' : (prefixes[f.fileCategory] || 'file');
for (const idx of [19, 20, 21, 22]) {
  const b = batches.batches.find(x => x.batchIndex === idx);
  if (!b) continue;
  const files = (b.files || []).map(x => typeof x === 'string' ? fileByPath.get(x) : fileByPath.get(x.path)).filter(Boolean);
  const nodes = files.map(f => ({ id: idFor(f), type: typeFor(f), name: path.basename(f.path), filePath: f.path, summary: summaryFor(f), tags: tagsFor(f), complexity: f.sizeLines > 200 ? 'complex' : f.sizeLines > 50 ? 'moderate' : 'simple' }));
  const known = new Set(nodes.map(n => n.id));
  const edges = [];
  for (const f of files) {
    const source = idFor(f);
    for (const targetPath of imports[f.path] || []) {
      const targetFile = fileByPath.get(targetPath);
      if (!targetFile) continue;
      edges.push({ source, target: idFor(targetFile), type: 'imports', direction: 'forward', weight: 0.7 });
    }
  }
  if (idx === 22) {
    const readme = 'document:README.md';
    if (known.has(readme)) {
      for (const n of nodes.filter(n => n.type === 'file' && (n.filePath.startsWith('example/') || n.filePath.startsWith('tools/')))) edges.push({ source: readme, target: n.id, type: 'documents', direction: 'forward', weight: 0.5 });
    }
  }
  fs.writeFileSync(path.join(inter, `batch-${idx}.json`), JSON.stringify({ nodes, edges }, null, 2) + '\n');
  console.log(`batch-${idx}: ${nodes.length} nodes, ${edges.length} edges`);
}
