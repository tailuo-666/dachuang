"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { MessageSquare, Settings, Paperclip, ImageIcon, Mic, Send, Plus, Sparkles, X, Trash2 } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  attachments?: Array<{
    type: "image" | "file" | "audio"
    name: string
    url: string
  }>
  metadata?: {
    questionId?: string
    retrievalTime?: number
    generationTime?: number
    confidence?: number
    sources?: string[]
  }
}

type Conversation = {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

export function MultimodalChat() {
  const [activeView, setActiveView] = useState<"chat" | "settings">("chat")

  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: "1",
      title: "当前对话",
      messages: [
        {
          id: "1",
          role: "assistant",
          content: "您好！我是多模态AI助手，可以处理文本、图片、文件和语音。请问有什么可以帮助您的？",
        },
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ])
  const [currentConversationId, setCurrentConversationId] = useState("1")

  const currentConversation = conversations.find((c) => c.id === currentConversationId)
  const messages = currentConversation?.messages || []

  const [inputValue, setInputValue] = useState("")
  const [attachments, setAttachments] = useState<
    Array<{
      type: "image" | "file" | "audio"
      name: string
      url: string
    }>
  >([])
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [isLoading, setIsLoading] = useState(false)

  // 模型设置状态
  const [streamEnabled, setStreamEnabled] = useState(true)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [maxTokensEnabled, setMaxTokensEnabled] = useState(false)

  // 流式输出状态
  const [streamingContent, setStreamingContent] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)

  // 将流式内容实时写入最后一条助手消息
  useEffect(() => {
    if (!isStreaming || !streamingContent) return
    setConversations((prev) =>
      prev.map((conv) => {
        if (conv.id !== currentConversationId) return conv
        const lastMsg = conv.messages.at(-1)
        if (!lastMsg || lastMsg.role !== "assistant") return conv
        return {
          ...conv,
          messages: [
            ...conv.messages.slice(0, -1),
            { ...lastMsg, content: streamingContent },
          ],
          updatedAt: new Date(),
        }
      }),
    )
  }, [streamingContent, isStreaming, currentConversationId])

  const fileInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const handleSend = async () => {
    if (!inputValue.trim() && attachments.length === 0) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
      attachments: attachments.length > 0 ? [...attachments] : undefined,
    }

    // 更新当前对话的消息
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === currentConversationId
          ? {
              ...conv,
              messages: [...conv.messages, userMessage],
              title: conv.messages.length === 1 ? inputValue.slice(0, 30) : conv.title,
              updatedAt: new Date(),
            }
          : conv,
      ),
    )

    const currentInput = inputValue
    setInputValue("")
    setAttachments([])
    setIsLoading(true)

    try {
      // 1. 调用用户提问接口
      console.log("[v0] 发送用户提问...")
      const questionResponse = await fetch("/api/chat/question", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: currentInput,
          conversationId: currentConversationId,
          attachments: attachments.map((a) => ({ type: a.type, name: a.name })),
          stream: streamEnabled,
          maxTokens: maxTokensEnabled ? maxTokens : undefined,
        }),
      })
      const questionData = await questionResponse.json()
      console.log("[v0] 用户提问响应:", questionData)

      // 2. 调用向量检索接口
      console.log("[v0] 开始向量检索...")
      const retrieveResponse = await fetch("/api/chat/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questionId: questionData.data.questionId,
          query: currentInput,
        }),
      })
      const retrieveData = await retrieveResponse.json()
      console.log("[v0] 向量检索响应:", retrieveData)

      // 3. 调用答案生成接口
      console.log("[v0] 生成答案...")
      const generateResponse = await fetch("/api/chat/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questionId: questionData.data.questionId,
          query: currentInput,
          retrievedDocs: retrieveData.data.results,
          conversationHistory: messages.slice(-5),
        }),
      })
      const generateData = await generateResponse.json()
      console.log("[v0] 答案生成响应:", generateData)

      // 4. 添加AI回复消息
      if (streamEnabled) {
        // 流式：先插入占位，后续逐字填充
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "", // 初始为空
          metadata: {
            questionId: questionData.data.questionId,
            retrievalTime: retrieveData.data.retrievalTime,
            generationTime: generateData.data.generationTime,
            confidence: generateData.data.confidence,
            sources: generateData.data.sources,
          },
        }
        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === currentConversationId
              ? {
                  ...conv,
                  messages: [...conv.messages, assistantMessage],
                  updatedAt: new Date(),
                }
              : conv,
          ),
        )
        // 启动流式读取
        await startStreaming(questionData.data.questionId)
      } else {
        // 非流式：直接填充完整内容
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: generateData.data.answer,
          metadata: {
            questionId: questionData.data.questionId,
            retrievalTime: retrieveData.data.retrievalTime,
            generationTime: generateData.data.generationTime,
            confidence: generateData.data.confidence,
            sources: generateData.data.sources,
          },
        }
        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === currentConversationId
              ? {
                  ...conv,
                  messages: [...conv.messages, assistantMessage],
                  updatedAt: new Date(),
                }
              : conv,
          ),
        )
      }
    } catch (error) {
      console.error("[v0] API 调用失败:", error)
      // 添加错误消息
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === currentConversationId
            ? {
                ...conv,
                messages: [
                  ...conv.messages,
                  {
                    id: (Date.now() + 1).toString(),
                    role: "assistant",
                    content: "抱歉，处理您的请求时出现了错误。请稍后重试。",
                  },
                ],
                updatedAt: new Date(),
              }
            : conv,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  const createNewConversation = () => {
    const currentConv = conversations.find((c) => c.id === currentConversationId)
    if (currentConv) {
      const hasUserMessages = currentConv.messages.some((msg) => msg.role === "user")
      if (!hasUserMessages) {
        // 用户还没有发送任何消息，不创建新对话
        return
      }
    }

    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: "新对话",
      messages: [
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "您好！我是多模态AI助手，可以处理文本、图片、文件和语音。请问有什么可以帮助您的？",
        },
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
    }
    setConversations((prev) => [newConversation, ...prev])
    setCurrentConversationId(newConversation.id)
  }

  const switchConversation = (conversationId: string) => {
    setCurrentConversationId(conversationId)
  }

  const deleteConversation = (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (conversations.length === 1) {
      alert("至少需要保留一个对话")
      return
    }
    setConversations((prev) => prev.filter((c) => c.id !== conversationId))
    if (currentConversationId === conversationId) {
      setCurrentConversationId(conversations.find((c) => c.id !== conversationId)?.id || conversations[0].id)
    }
  }

  const handleFileSelect = (type: "file" | "image") => {
    if (type === "file") {
      fileInputRef.current?.click()
    } else {
      imageInputRef.current?.click()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: "file" | "image") => {
    const files = e.target.files
    if (files && files[0]) {
      const file = files[0]
      setAttachments([
        ...attachments,
        {
          type: type === "image" ? "image" : "file",
          name: file.name,
          url: URL.createObjectURL(file),
        },
      ])
    }
  }

  const removeAttachment = (index: number) => {
    setAttachments(attachments.filter((_, i) => i !== index))
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" })
        const audioUrl = URL.createObjectURL(audioBlob)
        const timestamp = new Date().toLocaleTimeString("zh-CN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })

        setAttachments([
          ...attachments,
          {
            type: "audio",
            name: `录音 ${timestamp}.webm`,
            url: audioUrl,
          },
        ])

        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
    } catch (error) {
      console.error("无法访问麦克风:", error)
      alert("无法访问麦克风，请检查浏览器权限设置")
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)

      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current)
        recordingIntervalRef.current = null
      }
    }
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  const formatRecordingTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  // 流式输出：逐字填充最后一条助手消息
  const startStreaming = async (questionId: string) => {
    setIsStreaming(true)
    setStreamingContent("")
    try {
      const res = await fetch("/api/chat/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questionId,
          stream: true, // 关键标识
        }),
      })
      if (!res.ok || !res.body) throw new Error("流式响应异常")
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let done = false
      while (!done) {
        const { value, done: doneReading } = await reader.read()
        done = doneReading
        const chunk = decoder.decode(value, { stream: true })
        setStreamingContent((prev) => prev + chunk)
      }
    } catch (err) {
      console.error("[v0] 流式读取失败", err)
    } finally {
      setIsStreaming(false)
    }
  }

  return (
    <div className="flex h-screen max-h-screen bg-background overflow-hidden">
      {/* 左侧导航栏 */}
      <aside className="w-72 border-r border-border bg-sidebar flex flex-col shadow-sm">
        <div className="p-5 border-b border-sidebar-border">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-md">
              <Sparkles className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="font-semibold text-lg text-sidebar-foreground">多模态AI</h1>
              <p className="text-xs text-muted-foreground">智能对话助手</p>
            </div>
          </div>
          <Button
            onClick={createNewConversation}
            className="w-full justify-start gap-2 bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm"
            size="sm"
          >
            <Plus className="w-4 h-4" />
            新建对话
          </Button>
        </div>

        <ScrollArea className="flex-1 p-4">
          <div className="space-y-2">
            {conversations
              .slice()
              .sort((a, b) => {
                const aNew = a.messages.every((m) => m.role !== "user")
                const bNew = b.messages.every((m) => m.role !== "user")
                if (a.id === currentConversationId && aNew) return -1
                if (b.id === currentConversationId && bNew) return 1
                return b.updatedAt.getTime() - a.updatedAt.getTime()
              })
              .map((conversation) => (
                <div
                  key={conversation.id}
                  onClick={() => switchConversation(conversation.id)}
                  className={cn(
                    "px-4 py-3 rounded-xl cursor-pointer transition-all duration-200 group relative",
                    currentConversationId === conversation.id
                      ? "bg-primary/10 border border-primary/20 shadow-sm"
                      : "hover:bg-sidebar-accent border border-transparent",
                  )}
                >
                  <div className="flex items-start gap-2 mb-1">
                    <MessageSquare
                      className={cn(
                        "w-4 h-4 mt-0.5 flex-shrink-0",
                        currentConversationId === conversation.id ? "text-primary" : "text-muted-foreground",
                      )}
                    />
                    <span
                      className={cn(
                        "text-sm font-medium flex-1 line-clamp-1",
                        currentConversationId === conversation.id ? "text-foreground" : "text-sidebar-foreground",
                      )}
                    >
                      {conversation.title}
                    </span>
                    <button
                      onClick={(e) => deleteConversation(conversation.id, e)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-destructive/10 rounded"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-destructive" />
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground line-clamp-2 pl-6">
                    {conversation.messages[conversation.messages.length - 1]?.content || "暂无消息"}
                  </p>
                  <p className="text-xs text-muted-foreground/70 mt-1 pl-6">
                    {conversation.updatedAt.toLocaleString("zh-CN", {
                      month: "numeric",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              ))}
          </div>
        </ScrollArea>

        <div className="p-4 border-t border-sidebar-border space-y-2">
          <button
            onClick={() => setActiveView("chat")}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-sidebar-foreground",
              activeView === "chat" ? "bg-primary text-primary-foreground shadow-md" : "hover:bg-sidebar-accent",
            )}
          >
            <MessageSquare className="w-4 h-4" />
            <span className="text-sm font-medium">对话</span>
          </button>
          <button
            onClick={() => setActiveView("settings")}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-sidebar-foreground",
              activeView === "settings" ? "bg-primary text-primary-foreground shadow-md" : "hover:bg-sidebar-accent",
            )}
          >
            <Settings className="w-4 h-4" />
            <span className="text-sm font-medium">设置</span>
          </button>
        </div>
      </aside>

      {/* 主内容区域 */}
      <main className="flex-1 flex flex-col min-w-0">
        {activeView === "chat" ? (
          <>
            {/* 聊天消息区域 */}
            <ScrollArea className="flex-1 px-6 py-8">
              <div className="max-w-4xl mx-auto space-y-8">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={cn(
                      "flex gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500",
                      message.role === "user" ? "justify-end" : "justify-start",
                    )}
                  >
                    {message.role === "assistant" && (
                      <Avatar className="w-9 h-9 border-2 border-primary/20 shadow-sm">
                        <AvatarFallback className="bg-gradient-to-br from-primary to-accent text-primary-foreground text-sm font-medium">
                          AI
                        </AvatarFallback>
                      </Avatar>
                    )}
                    <div
                      className={cn(
                        "rounded-2xl px-5 py-3.5 max-w-[75%] shadow-sm",
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-card border border-border text-card-foreground",
                      )}
                    >
                      {message.attachments && message.attachments.length > 0 && (
                        <div className="mb-3 space-y-2">
                          {message.attachments.map((attachment, idx) => (
                            <div key={idx}>
                              {attachment.type === "image" ? (
                                <div className="rounded-lg overflow-hidden border border-border/50">
                                  <img
                                    src={attachment.url || "/placeholder.svg"}
                                    alt={attachment.name}
                                    className="max-w-full h-auto max-h-64 object-cover"
                                  />
                                  <div
                                    className={cn(
                                      "flex items-center gap-2 p-2 text-xs",
                                      message.role === "user" ? "bg-primary-foreground/15" : "bg-muted/50",
                                    )}
                                  >
                                    <ImageIcon className="w-3.5 h-3.5" />
                                    <span className="truncate">{attachment.name}</span>
                                  </div>
                                </div>
                              ) : attachment.type === "audio" ? (
                                <div
                                  className={cn(
                                    "rounded-lg overflow-hidden border border-border/50",
                                    message.role === "user" ? "bg-primary-foreground/15" : "bg-muted/50",
                                  )}
                                >
                                  <audio controls className="w-full h-10">
                                    <source src={attachment.url} type="audio/webm" />
                                    您的浏览器不支持音频播放
                                  </audio>
                                  <div className="flex items-center gap-2 p-2 text-xs border-t border-border/30">
                                    <Mic className="w-3.5 h-3.5" />
                                    <span className="truncate">{attachment.name}</span>
                                  </div>
                                </div>
                              ) : (
                                <div
                                  className={cn(
                                    "flex items-center gap-2 p-2.5 rounded-lg",
                                    message.role === "user" ? "bg-primary-foreground/15" : "bg-muted/50",
                                  )}
                                >
                                  <Paperclip className="w-4 h-4" />
                                  <span className="text-sm truncate">{attachment.name}</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      <p className="text-sm leading-relaxed text-pretty whitespace-pre-wrap">
                        {message.role === "assistant" &&
                        idx === messages.length - 1 &&
                        isStreaming
                          ? streamingContent
                          : message.content}
                      </p>
                      {message.metadata && message.role === "assistant" && (
                        <div className="mt-3 pt-3 border-t border-border/30 text-xs text-muted-foreground space-y-1">
                          {message.metadata.confidence && (
                            <div>置信度: {(message.metadata.confidence * 100).toFixed(0)}%</div>
                          )}
                          {message.metadata.retrievalTime && (
                            <div>检索耗时: {message.metadata.retrievalTime.toFixed(2)}s</div>
                          )}
                          {message.metadata.generationTime && (
                            <div>生成耗时: {message.metadata.generationTime.toFixed(2)}s</div>
                          )}
                          {message.metadata.sources && message.metadata.sources.length > 0 && (
                            <div>来源: {message.metadata.sources.join(", ")}</div>
                          )}
                        </div>
                      )}
                    </div>
                    {message.role === "user" && (
                      <Avatar className="w-9 h-9 border-2 border-secondary shadow-sm">
                        <AvatarFallback className="bg-secondary text-secondary-foreground text-sm font-medium">
                          我
                        </AvatarFallback>
                      </Avatar>
                    )}
                  </div>
                ))}
                {(isLoading || isStreaming) && (
                  <div className="flex gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <Avatar className="w-9 h-9 border-2 border-primary/20 shadow-sm">
                      <AvatarFallback className="bg-gradient-to-br from-primary to-accent text-primary-foreground text-sm font-medium">
                        AI
                      </AvatarFallback>
                    </Avatar>
                    <div className="rounded-2xl px-5 py-3.5 bg-card border border-border">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-primary animate-bounce" />
                        <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.2s]" />
                        <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.4s]" />
                        <span className="text-sm text-muted-foreground ml-2">{isStreaming ? "AI正在输出..." : "AI正在思考..."}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* 输入区域 */}
            <div className="border-t border-border bg-card/50 backdrop-blur-sm">
              <div className="max-w-4xl mx-auto p-6">
                {isRecording && (
                  <div className="mb-4 flex items-center gap-3 px-4 py-3 bg-destructive/10 border border-destructive/20 rounded-xl animate-in fade-in slide-in-from-bottom-2">
                    <div className="w-3 h-3 rounded-full bg-destructive animate-pulse" />
                    <span className="text-sm font-medium text-destructive">正在录音...</span>
                    <span className="text-sm text-muted-foreground ml-auto">{formatRecordingTime(recordingTime)}</span>
                  </div>
                )}

                {attachments.length > 0 && (
                  <div className="mb-4 flex flex-wrap gap-2">
                    {attachments.map((attachment, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-2 px-3 py-2 bg-secondary/80 rounded-xl text-sm shadow-sm animate-in fade-in zoom-in-95 duration-200"
                      >
                        {attachment.type === "image" ? (
                          <>
                            <div className="w-8 h-8 rounded overflow-hidden border border-border/50">
                              <img
                                src={attachment.url || "/placeholder.svg"}
                                alt={attachment.name}
                                className="w-full h-full object-cover"
                              />
                            </div>
                            <span className="truncate max-w-[120px]">{attachment.name}</span>
                          </>
                        ) : attachment.type === "audio" ? (
                          <>
                            <Mic className="w-4 h-4 text-primary" />
                            <span className="truncate max-w-[150px]">{attachment.name}</span>
                          </>
                        ) : (
                          <>
                            <Paperclip className="w-4 h-4 text-primary" />
                            <span className="truncate max-w-[150px]">{attachment.name}</span>
                          </>
                        )}
                        <button
                          onClick={() => removeAttachment(idx)}
                          className="hover:bg-background rounded-lg p-1 transition-colors"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="flex items-end gap-3">
                  <div className="flex-1 relative">
                    <Input
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault()
                          handleSend()
                        }
                      }}
                      placeholder={isStreaming ? "AI正在输出中，请稍候..." : "输入消息或添加文件、图片、录音..."}
                      className="pr-36 min-h-[52px] resize-none bg-background border-border shadow-sm rounded-xl text-sm"
                      disabled={isRecording || isLoading || isStreaming}
                    />
                    <div className="absolute right-2 bottom-2 flex items-center gap-1">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-9 w-9 hover:bg-accent"
                        onClick={() => handleFileSelect("file")}
                        disabled={isRecording || isLoading || isStreaming}
                      >
                        <Paperclip className="w-4 h-4" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-9 w-9 hover:bg-accent"
                        onClick={() => handleFileSelect("image")}
                        disabled={isRecording || isLoading || isStreaming}
                      >
                        <ImageIcon className="w-4 h-4" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className={cn(
                          "h-9 w-9 transition-colors",
                          isRecording
                            ? "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            : "hover:bg-accent",
                        )}
                        onClick={toggleRecording}
                        disabled={isLoading || isStreaming}
                      >
                        <Mic className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  <Button
                    onClick={handleSend}
                    size="icon"
                    className="h-[52px] w-[52px] flex-shrink-0 rounded-xl shadow-md hover:shadow-lg transition-shadow"
                    disabled={isRecording || isLoading || isStreaming}
                  >
                    <Send className="w-5 h-5" />
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground mt-3 text-center">
                  支持文本、图片、文件和语音等多种输入方式
                </p>
              </div>
            </div>

            {/* 隐藏的文件输入 */}
            <input ref={fileInputRef} type="file" className="hidden" onChange={(e) => handleFileChange(e, "file")} />
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFileChange(e, "image")}
            />
          </>
        ) : (
          <div className="flex-1 px-6 py-8 overflow-y-auto">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-3xl font-semibold mb-8 text-balance">设置</h2>
              <div className="space-y-6">
                <div className="bg-card border border-border rounded-2xl p-6 shadow-sm">
                  <h3 className="font-semibold text-lg mb-5 flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary"></div>
                    模型设置
                  </h3>
                  <div className="space-y-5">
                    <div>
                      <label className="text-sm font-medium text-foreground mb-2 block">AI模型</label>
                      <select className="w-full px-4 py-2.5 bg-background border border-border rounded-xl text-sm focus:ring-2 focus:ring-primary/20 transition-shadow">
                        <option>GPT-4 Turbo</option>
                        <option>Claude 3 Opus</option>
                        <option>Gemini Pro</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-foreground mb-2 block">温度</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        defaultValue="0.7"
                        className="w-full accent-primary"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>精确</span>
                        <span>创造</span>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <label className="text-sm font-medium text-foreground">流式输入</label>
                        <Switch
                          checked={streamEnabled}
                          onCheckedChange={setStreamEnabled}
                        />
                      </div>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <label className="text-sm font-medium text-foreground">最大 token 数</label>
                          <Switch
                            checked={maxTokensEnabled}
                            onCheckedChange={setMaxTokensEnabled}
                          />
                        </div>
                        {maxTokensEnabled && (
                          <div className="flex items-center gap-3">
                            <Input
                              type="number"
                              min={1}
                              max={8192}
                              value={maxTokens}
                              onChange={(e) => setMaxTokens(Number(e.target.value))}
                              className="w-24 text-sm"
                            />
                            <span className="text-xs text-muted-foreground">1–8192</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>


              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
