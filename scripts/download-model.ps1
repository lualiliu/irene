# 下载向量模型到 docs/models（Invoke-WebRequest 会走系统代理，Node 的 fetch 不会）
$base = 'https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main'
$dst = Join-Path $PSScriptRoot '..\docs\models\Xenova\bge-small-zh-v1.5'
New-Item -ItemType Directory -Force (Join-Path $dst 'onnx') | Out-Null
foreach ($f in 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'onnx/model_quantized.onnx') {
    Invoke-WebRequest "$base/$f" -UseBasicParsing -OutFile (Join-Path $dst $f)
    Write-Host "已下载 $f"
}
