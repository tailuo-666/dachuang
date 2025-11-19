import { type NextRequest, NextResponse } from "next/server"

// 爬虫触发接口
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { url, depth = 1, maxPages = 10 } = body

    console.log("[v0] 爬虫触发接口收到请求:", { url, depth, maxPages })

    // 模拟爬虫启动延迟
    await new Promise((resolve) => setTimeout(resolve, 600))

    // Mock 爬虫任务数据
    const taskId = `crawler_${Date.now()}`

    const response = {
      success: true,
      data: {
        taskId,
        url,
        status: "running",
        config: {
          depth,
          maxPages,
        },
        startTime: new Date().toISOString(),
        estimatedTime: 30, // 秒
      },
      message: "爬虫任务已启动",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] 爬虫触发接口错误:", error)
    return NextResponse.json({ success: false, message: "启动失败" }, { status: 500 })
  }
}

// 查询爬虫任务状态
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const taskId = searchParams.get("taskId")

    console.log("[v0] 查询爬虫状态:", taskId)

    // Mock 爬虫状态数据
    const response = {
      success: true,
      data: {
        taskId,
        status: "completed",
        progress: 100,
        pagesScraped: 8,
        documentsCreated: 15,
        completedTime: new Date().toISOString(),
        results: [
          { url: "https://example.com/page1", title: "页面1", status: "success" },
          { url: "https://example.com/page2", title: "页面2", status: "success" },
          { url: "https://example.com/page3", title: "页面3", status: "success" },
        ],
      },
      message: "任务已完成",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] 查询爬虫状态错误:", error)
    return NextResponse.json({ success: false, message: "查询失败" }, { status: 500 })
  }
}
