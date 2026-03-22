from flask import Flask, request, jsonify, send_file, render_template_string, Response  # 新增Response导入
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import json
import io

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# 全局变量
groups = []
capacity = 0
best_value = 0
solve_time = 0
dp_history = []
# 新增：存储排序结果详情
sort_details = []

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# 读取数据
def read_data(file_path):
    global groups, capacity, sort_details
    groups = []
    capacity = 0
    sort_details = []  # 重置排序详情
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if len(lines) == 0:
            return False
        count, capacity = map(int, lines[0].split())
        idx = 1
        # 校验数据行数是否足够
        if len(lines) < 1 + count * 3:
            return False
        for _ in range(count):
            i1 = list(map(int, lines[idx].split()))
            i2 = list(map(int, lines[idx + 1].split()))
            i3 = list(map(int, lines[idx + 2].split()))
            groups.append([i1, i2, i3])
            idx += 3
        return True
    except Exception as e:
        print(f"读取数据错误: {e}")
        return False


# 按第三项价值/重量比降序排序（改造：返回排序详情）
def sort_data():
    global groups, sort_details
    sort_details = []  # 清空历史排序详情
    if not groups:
        return False, []

    # 先记录每组的原始信息（组号、第三项重量、第三项价值、价值/重量比）
    for idx, g in enumerate(groups):
        w3, v3 = g[2]
        ratio = (v3 / w3) if w3 != 0 else 0
        sort_details.append({
            "original_group": idx + 1,  # 原始组号（从1开始）
            "weight_3": w3,
            "value_3": v3,
            "ratio": round(ratio, 4)  # 保留4位小数
        })

    # 按价值/重量比降序排序（同时记录排序依据）
    sorted_with_info = sorted(
        enumerate(groups),
        key=lambda x: (x[1][2][1] / x[1][2][0]) if x[1][2][0] != 0 else 0,
        reverse=True
    )

    # 更新groups为排序后的结果
    groups = [item[1] for item in sorted_with_info]

    # 整理排序后的详情（包含排序后顺序）
    sorted_details = []
    for new_idx, (old_idx, _) in enumerate(sorted_with_info):
        detail = sort_details[old_idx]
        detail["sorted_order"] = new_idx + 1  # 排序后的顺序（从1开始）
        sorted_details.append(detail)

    sort_details = sorted_details
    return True, sorted_details


# 动态规划求解 + 记录每一步DP状态（用于动画）
def dp_solve():
    global best_value, solve_time, dp_history
    if not groups or capacity == 0:
        return 0, 0
    dp = [0] * (capacity + 1)
    dp_history = []  # 记录每一组处理后的DP数组
    start = time.time()

    # 处理每一组物品，记录状态
    for idx, g in enumerate(groups):
        w1, v1 = g[0]
        w2, v2 = g[1]
        w3, v3 = g[2]
        tmp = dp.copy()

        # 逆序更新DP数组
        for j in range(capacity, -1, -1):
            if j >= w1:
                tmp[j] = max(tmp[j], dp[j - w1] + v1)
            if j >= w2:
                tmp[j] = max(tmp[j], dp[j - w2] + v2)
            if j >= w3:
                tmp[j] = max(tmp[j], dp[j - w3] + v3)

        dp = tmp
        dp_history.append(dp.copy())  # 保存当前组处理后的状态

    best_value = dp[capacity]
    solve_time = round(time.time() - start, 4)
    return best_value, solve_time


# 绘制指定组的散点图（核心新增功能）
def draw_group_scatter(group_num):
    """绘制指定组的重量-价值散点图"""
    # 完善数据校验
    if not groups:
        return None
    if group_num < 1 or group_num > len(groups):
        return None

    try:
        # 获取指定组的数据
        group_data = groups[group_num - 1]
        weights = [item[0] for item in group_data]
        values = [item[1] for item in group_data]

        # 处理空数据/零值情况
        if len(weights) == 0 or max(weights) == 0:
            weights = [1, 2, 3]  # 兜底值
        if len(values) == 0 or max(values) == 0:
            values = [1, 2, 3]  # 兜底值

        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 6))
        # 绘制散点（每个点标注物品编号）
        for i, (w, v) in enumerate(zip(weights, values)):
            ax.scatter(w, v, c="#6a11cb", s=150, alpha=0.8, edgecolors="white", zorder=5)
            ax.annotate(f"物品{i + 1}", (w, v), xytext=(5, 5), textcoords="offset points", fontsize=12)

        # 设置图表样式
        ax.set_xlabel("重量", fontsize=14)
        ax.set_ylabel("价值", fontsize=14)
        ax.set_title(f"D{{0-1}}KP 第{group_num}组数据 - 重量-价值散点图", fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlim(0, max(weights) + 1)
        ax.set_ylim(0, max(values) + 1)

        # 保存到内存（避免文件IO）
        img_buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)

        return img_buffer
    except Exception as e:
        print(f"绘制散点图错误: {e}")
        return None


# 绘制全部数据的散点图
def draw_all_scatter():
    try:
        if not groups:
            return False
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ws, vs = [], []
        for g in groups:
            for it in g:
                ws.append(it[0])
                vs.append(it[1])

        # 兜底处理空数据
        if len(ws) == 0:
            ws = [1, 2, 3]
        if len(vs) == 0:
            vs = [1, 2, 3]

        ax.scatter(ws, vs, c="#6a11cb", s=70, alpha=0.8, edgecolors="white")
        ax.set_xlabel("重量")
        ax.set_ylabel("价值")
        ax.set_title("D{0-1}KP 全部数据 - 重量-价值散点图")
        ax.grid(alpha=0.3, linestyle="--")
        plt.savefig('static/plot.png', bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"绘制全部散点图错误: {e}")
        return False


# 主页面（含任意组散点图功能）
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>D{0-1}背包求解系统【全能可视化版】</title>
    <style>
        * {margin: 0; padding: 0; box-sizing: border-box; font-family: "Microsoft YaHei", sans-serif;}
        body {background: linear-gradient(135deg, #f0f4ff, #e2eaff); padding: 30px;}
        .container {max-width: 1200px; margin: 0 auto; background: white; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.08);}
        .header {background: linear-gradient(90deg, #6a11cb, #2575fc); color: white; padding: 30px; text-align: center;}
        .header h1 {font-size: 28px; margin-bottom: 8px;}
        .header p {font-size: 16px; opacity: 0.9;}
        .panel {padding: 24px; background: #f9fbff; border-radius: 16px; margin: 20px;}
        .btn {
            padding: 14px 24px; border: none; border-radius: 12px; 
            background: #6a11cb; color: white; cursor: pointer; 
            margin: 8px; transition: 0.3s; font-size: 15px;
        }
        .btn:hover {
            transform: translateY(-3px); 
            box-shadow: 0 5px 15px rgba(106,17,203,0.3);
        }
        .btn-green {background: #28a745;}
        .btn-blue {background: #007bff;}
        #result {
            padding: 20px; background: #effffa; border-radius: 12px; 
            line-height: 1.8; font-size: 16px; margin: 10px 0;
        }
        #dpCanvas {
            border-radius: 12px; margin-top: 20px; 
            background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            width: 100%; height: 400px;
        }
        .loading {color: #6a11cb; font-weight: bold;}
        .group-select {
            padding: 10px; border: 1px solid #ddd; border-radius: 8px; 
            margin: 8px; width: 200px; font-size: 15px;
        }
        .scatter-container {margin-top: 20px;}
        #scatterImg {
            width: 100%; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-top: 10px;
        }
        /* 新增：排序结果表格样式 */
        .sort-result-table {
            margin: 10px 0;
            border-collapse: collapse;
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .sort-result-table th, .sort-result-table td {
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .sort-result-table th {
            background: #6a11cb;
            color: white;
        }
        .sort-result-table tr:hover {
            background: #f8f9ff;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🎯 D{0-1}背包智能求解系统</h1>
    </div>

    <div class="panel">
        <!-- 基础操作区 -->
        <input type="file" id="file" accept=".txt" style="padding: 10px; margin: 8px 0;">
        <button class="btn" onclick="upload()">📂 上传数据</button>
        <button class="btn" onclick="sortData()">⚡ 按第三项价值密度排序</button>
        <button class="btn" onclick="solve()">💡 计算最优解</button>
        <button class="btn" onclick="playAnimation()">🎬 播放DP求解动画</button>
        <button class="btn btn-green" onclick="exportReport()">📄 导出报告</button>

        <!-- 新增：任意组散点图绘制区 -->
        <div style="margin: 15px 0; padding: 15px; background: #f0f8ff; border-radius: 10px;">
            <h3 style="color: #6a11cb; margin-bottom: 10px;">🎨 绘制指定组散点图</h3>
            <input type="number" id="groupNum" class="group-select" placeholder="输入组号（如1、2、3...）" min="1">
            <button class="btn btn-blue" onclick="drawGroupScatter()">🖌️ 绘制该组散点图</button>
            <button class="btn btn-blue" onclick="drawAllScatter()">🖌️ 绘制全部数据散点图</button>
        </div>

        <div id="result">等待操作...</div>

        <!-- DP动画画布 -->
        <div style="margin-top: 20px;">
            <h4>📈 DP状态转移动画</h4>
            <canvas id="dpCanvas"></canvas>
        </div>

        <!-- 散点图画布 -->
        <div class="scatter-container">
            <h4>📊 重量-价值散点图</h4>
            <img id="scatterImg" style="display: none;">
        </div>
    </div>
</div>

<script>
    // 全局变量
    let dpHistory = [];    // DP状态历史
    let maxValue = 0;      // 最大价值（用于缩放柱状图）
    let cap = 0;           // 背包容量
    let groupCount = 0;    // 项集总数
    let canvas = document.getElementById("dpCanvas");
    let ctx = canvas.getContext("2d");

    // 初始化画布大小
    function initCanvas() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    window.onresize = initCanvas;
    initCanvas();

    // 上传数据
    function upload() {
        let fileInput = document.getElementById("file");
        if (!fileInput.files[0]) {
            alert("请选择数据文件！");
            return;
        }
        let fd = new FormData();
        fd.append("file", fileInput.files[0]);
        fetch("/upload", { method: "POST", body: fd })
            .then(res => res.json())
            .then(res => {
                cap = res.cap;
                groupCount = res.groups;
                document.getElementById("result").innerHTML = 
                    "✅ 数据加载成功：<br>" + 
                    "项集数：" + res.groups + " 组 | 背包容量：" + res.cap + "<br>" +
                    "💡 可输入1-" + res.groups + "之间的组号绘制散点图";
            })
            .catch(err => alert("上传失败：" + err.message || "未知错误"));
    }

    // 排序（改造：接收并展示排序结果）
    function sortData() {
        fetch("/sort")
            .then(res => res.json())
            .then(res => {
                if (res.ok) {
                    // 基础提示文字
                    let baseText = "✅ 已按【第三项价值/重量比】降序排序完成！<br>项集总数：" + groupCount + " 组<br><br>";
                    // 拼接排序结果表格
                    let tableHtml = "<table class='sort-result-table'>" +
                        "<tr><th>排序后顺序</th><th>原始组号</th><th>第三项重量</th><th>第三项价值</th><th>价值/重量比</th></tr>";

                    // 遍历排序详情，生成表格行
                    res.details.forEach(item => {
                        tableHtml += `<tr>
                            <td>${item.sorted_order}</td>
                            <td>${item.original_group}</td>
                            <td>${item.weight_3}</td>
                            <td>${item.value_3}</td>
                            <td>${item.ratio}</td>
                        </tr>`;
                    });
                    tableHtml += "</table>";

                    // 最终展示内容
                    document.getElementById("result").innerHTML = baseText + tableHtml;
                } else {
                    alert("排序失败：暂无数据");
                    document.getElementById("result").innerHTML = "❌ 排序失败：暂无数据可排序！";
                }
            })
            .catch(err => {
                alert("排序失败：" + err.message || "未知错误");
                document.getElementById("result").innerHTML = "❌ 排序失败：" + (err.message || "未知错误");
            });
    }

    // 求解最优解
    function solve() {
        if (groupCount === 0) {
            alert("请先上传数据！");
            return;
        }
        document.getElementById("result").innerHTML = "<span class='loading'>🔄 正在计算最优解...</span>";
        fetch("/solve")
            .then(res => res.json())
            .then(res => {
                dpHistory = res.history || [];
                maxValue = res.value;
                document.getElementById("result").innerHTML = 
                    "🎯 最优求解完成：<br>" +
                    "🔹 最大价值：<b>" + res.value + "</b><br>" +
                    "🔹 求解耗时：" + res.time + "s<br>" +
                    "🔹 点击【播放DP求解动画】查看状态转移过程";
            })
            .catch(err => alert("求解失败：" + err.message || "未知错误"));
    }

    // ==================== DP动画播放 ====================
    function playAnimation() {
        if (dpHistory.length === 0) {
            alert("请先计算最优解！");
            return;
        }

        let step = 0;          // 当前播放到第几组
        const intervalTime = 800; // 每组动画间隔（毫秒）

        // 清空画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 开始逐组播放动画
        let animationInterval = setInterval(() => {
            // 播放完毕
            if (step >= dpHistory.length) {
                clearInterval(animationInterval);
                drawFinalState(); // 绘制最终状态
                return;
            }

            // 绘制当前组的DP状态
            drawDPState(step);
            step++;
        }, intervalTime);
    }

    // 绘制单步DP状态（柱状图）
    function drawDPState(step) {
        // 清空画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 获取当前组的DP数据
        let dpData = dpHistory[step];
        let barCount = dpData.length;
        let barWidth = canvas.width / barCount - 2; // 柱状图宽度（留间距）
        let canvasHeight = canvas.height - 60;     // 预留文字区域

        // 绘制标题
        ctx.fillStyle = "#333";
        ctx.font = "18px Microsoft YaHei";
        ctx.fillText(`DP状态转移动画 - 处理第 ${step+1}/${dpHistory.length} 组`, 20, 30);

        // 绘制坐标轴
        ctx.strokeStyle = "#999";
        ctx.beginPath();
        ctx.moveTo(20, canvasHeight + 20);
        ctx.lineTo(canvas.width - 20, canvasHeight + 20); // X轴
        ctx.moveTo(20, 20);
        ctx.lineTo(20, canvasHeight + 20); // Y轴
        ctx.stroke();

        // 绘制柱状图
        for (let i = 0; i < barCount; i++) {
            let value = dpData[i];
            // 计算柱状图高度（缩放适配画布）
            let barHeight = maxValue > 0 ? (value / maxValue) * canvasHeight : 0;
            let x = 20 + i * (barWidth + 2);
            let y = canvasHeight + 20 - barHeight;

            // 绘制柱子
            ctx.fillStyle = `rgba(106, 17, 203, ${0.6 + (step / dpHistory.length) * 0.4})`;
            ctx.fillRect(x, y, barWidth, barHeight);

            // 绘制边框
            ctx.strokeStyle = "#666";
            ctx.strokeRect(x, y, barWidth, barHeight);

            // 标注X轴（每5个标一次，避免拥挤）
            if (i % 5 === 0) {
                ctx.fillStyle = "#666";
                ctx.font = "12px Microsoft YaHei";
                ctx.fillText(i, x + barWidth/2 - 5, canvasHeight + 40);
            }
        }

        // 标注Y轴最大值
        ctx.fillStyle = "#666";
        ctx.font = "12px Microsoft YaHei";
        ctx.fillText(maxValue, 5, 30);
    }

    // 绘制最终状态
    function drawFinalState() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 绘制最终DP状态
        let finalData = dpHistory[dpHistory.length - 1];
        let barCount = finalData.length;
        let barWidth = canvas.width / barCount - 2;
        let canvasHeight = canvas.height - 60;

        // 标题
        ctx.fillStyle = "#28a745";
        ctx.font = "20px Microsoft YaHei";
        ctx.fillText(`DP求解完成 - 最终状态 (最大价值：${maxValue})`, 20, 30);

        // 坐标轴
        ctx.strokeStyle = "#999";
        ctx.beginPath();
        ctx.moveTo(20, canvasHeight + 20);
        ctx.lineTo(canvas.width - 20, canvasHeight + 20);
        ctx.moveTo(20, 20);
        ctx.lineTo(20, canvasHeight + 20);
        ctx.stroke();

        // 最终柱状图（绿色高亮）
        for (let i = 0; i < barCount; i++) {
            let value = finalData[i];
            let barHeight = maxValue > 0 ? (value / maxValue) * canvasHeight : 0;
            let x = 20 + i * (barWidth + 2);
            let y = canvasHeight + 20 - barHeight;

            // 最终柱子用绿色
            ctx.fillStyle = "rgba(40, 167, 69, 0.8)";
            ctx.fillRect(x, y, barWidth, barHeight);
            ctx.strokeStyle = "#28a745";
            ctx.strokeRect(x, y, barWidth, barHeight);

            // X轴标注
            if (i % 5 === 0) {
                ctx.fillStyle = "#666";
                ctx.font = "12px Microsoft YaHei";
                ctx.fillText(i, x + barWidth/2 - 5, canvasHeight + 40);
            }
        }

        // Y轴最大值
        ctx.fillStyle = "#666";
        ctx.font = "12px Microsoft YaHei";
        ctx.fillText(maxValue, 5, 30);
    }

    // ==================== 新增：绘制指定组散点图 ====================
    function drawGroupScatter() {
        let groupNum = document.getElementById("groupNum").value;
        if (!groupNum || groupNum < 1 || groupNum > groupCount) {
            alert(`请输入1-${groupCount}之间的有效组号！`);
            return;
        }

        document.getElementById("result").innerHTML = `<span class='loading'>🔄 正在绘制第${groupNum}组散点图...</span>`;

        fetch(`/draw-group-scatter?group=${groupNum}`)
            .then(res => {
                if (res.ok) return res.blob();
                throw new Error("绘制失败：" + res.statusText);
            })
            .then(blob => {
                let imgUrl = URL.createObjectURL(blob);
                let imgElement = document.getElementById("scatterImg");
                imgElement.src = imgUrl;
                imgElement.style.display = "block";
                document.getElementById("result").innerHTML = `✅ 第${groupNum}组散点图绘制完成！`;
            })
            .catch(err => {
                console.error("绘制散点图错误：", err);
                alert("绘制失败：" + err.message);
                document.getElementById("result").innerHTML = "❌ 散点图绘制失败，请检查组号或数据！";
            });
    }

    // 绘制全部数据散点图
    function drawAllScatter() {
        if (groupCount === 0) {
            alert("请先上传数据！");
            return;
        }
        document.getElementById("result").innerHTML = "<span class='loading'>🔄 正在绘制全部数据散点图...</span>";
        fetch("/draw-all-scatter")
            .then(res => res.json())
            .then(res => {
                if (res.ok) {
                    let imgElement = document.getElementById("scatterImg");
                    imgElement.src = "/static/plot.png?" + new Date().getTime(); // 加时间戳避免缓存
                    imgElement.style.display = "block";
                    document.getElementById("result").innerHTML = "✅ 全部数据散点图绘制完成！";
                } else {
                    throw new Error("绘制失败");
                }
            })
            .catch(err => {
                console.error("绘制全部散点图错误：", err);
                alert("绘制失败：" + err.message);
                document.getElementById("result").innerHTML = "❌ 全部数据散点图绘制失败！";
            });
    }

    // 导出报告
    function exportReport() {
        if (groupCount === 0) {
            alert("请先上传数据并求解！");
            return;
        }
        window.open("/export/full-report");
    }
</script>
</body>
</html>
    '''


# 接口：上传数据
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        path = "uploads/data.txt"
        file.save(path)
        success = read_data(path)
        if success:
            draw_all_scatter()
            return jsonify({"groups": len(groups), "cap": capacity})
        else:
            return jsonify({"groups": 0, "cap": 0, "error": "数据格式错误"}), 400
    except Exception as e:
        return jsonify({"groups": 0, "cap": 0, "error": str(e)}), 500


# 接口：排序（改造：返回排序详情）
@app.route('/sort')
def sort_route():
    success, details = sort_data()
    return jsonify({"ok": success, "details": details})


# 接口：求解并返回DP历史
@app.route('/solve')
def solve_route():
    v, t = dp_solve()
    return jsonify({
        "value": v,
        "time": t,
        "history": dp_history  # 返回每一步的DP状态
    })


# 新增接口：绘制指定组散点图
@app.route('/draw-group-scatter')
def draw_group_scatter_route():
    try:
        group_num = int(request.args.get('group', 1))
        img_buffer = draw_group_scatter(group_num)
        if not img_buffer:
            return Response("无效的组号或数据错误", status=400)
        return Response(img_buffer, mimetype='image/png')
    except Exception as e:
        print(f"散点图接口错误: {e}")
        return Response(f"绘制失败：{str(e)}", status=500)


# 接口：绘制全部数据散点图
@app.route('/draw-all-scatter')
def draw_all_scatter_route():
    success = draw_all_scatter()
    return jsonify({"ok": success})


# 接口：导出完整报告
@app.route('/export/full-report')
def export_full():
    dp_solve()
    # 补充排序结果到报告中
    sort_report = ""
    if sort_details:
        sort_report = "\n📊 排序详情（按第三项价值/重量比降序）：\n"
        for item in sort_details:
            sort_report += f"- 排序后第{item['sorted_order']}位：原始组{item['original_group']} | 重量{item['weight_3']} | 价值{item['value_3']} | 比值{item['ratio']}\n"

    report_content = f"""
D{0 - 1}背包问题求解报告
=====================================
📋 基础信息：
- 背包容量：{capacity}
- 项集数量：{len(groups)}
- 最优价值：{best_value}
- 求解耗时：{solve_time}s

🔍 排序规则：
按每个项集第三项的「价值/重量比」降序排列
{sort_report}

🎯 求解算法：
动态规划（DP）算法，记录每一步状态转移过程并可视化
"""
    with open("D01背包求解报告.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    return send_file("D01背包求解报告.txt", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)