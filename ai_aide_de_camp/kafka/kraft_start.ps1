# 配置 Kafka 路径
$kafka_home = "B:\kafka"
$config = "$kafka_home\config\kraft\server.properties"
$tmp_folder = "B:\tmp\kafka-logs"
$cluster_id_file = "$kafka_home\cluster.id.txt"

# 清理环境（首次启动时会删除旧日志目录）
if (-Not (Test-Path "$tmp_folder\meta.properties")) {
    Remove-Item -Path $tmp_folder -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -Path $tmp_folder -ItemType Directory | Out-Null
}

# 获取或生成 cluster.id
if (Test-Path $cluster_id_file) {
    # 如果已存在 cluster.id，则读取
    $cluster_id = Get-Content $cluster_id_file
} else {
    # 如果不存在，则生成新的 cluster.id 并保存
    $cluster_id = & "$kafka_home\bin\windows\kafka-storage.bat" random-uuid
    $cluster_id | Set-Content $cluster_id_file
}

# 更新配置文件中的 cluster.id
(Get-Content $config) -replace 'cluster.id=.*', "cluster.id=$cluster_id" | Set-Content $config

# 初始化存储（仅在日志目录不存在 meta.properties 时进行格式化）
if (-Not (Test-Path "$tmp_folder\meta.properties")) {
    & "$kafka_home\bin\windows\kafka-storage.bat" format -t $cluster_id -c $config
}

# 启动 Kafka 服务
Start-Process -FilePath "$kafka_home\bin\windows\kafka-server-start.bat" -ArgumentList $config -WindowStyle Hidden

# 健康检查：检查端口是否打开
$port_open = $false
1..30 | ForEach-Object {
    try {
        $socket = New-Object System.Net.Sockets.TcpClient("localhost", 9092)
        $socket.Close()
        $port_open = $true
        return
    } catch {}
    Start-Sleep -Seconds 2
}
if ($port_open) {
    Write-Host "Kafka 启动成功！" -ForegroundColor Green
} else {
    Write-Host "Kafka 启动失败，请检查日志。" -ForegroundColor Red
}
