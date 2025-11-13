import { type NextRequest, NextResponse } from "next/server"

// 用户提问接口
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { message, conversationId, attachments } = body

    console.log("[v0] 用户提问接口收到请求:", { message, conversationId, attachments })

    // 模拟处理延迟
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Mock 响应数据
    const response = {
      success: true,
      data: {
        questionId: `q_${Date.now()}`,
        conversationId: conversationId || `conv_${Date.now()}`,
        message,
        timestamp: new Date().toISOString(),
        attachments: attachments || [],
      },
      message: "问题已接收",
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] 用户提问接口错误:", error)
    return NextResponse.json({ success: false, message: "处理失败" }, { status: 500 })
  }
}
