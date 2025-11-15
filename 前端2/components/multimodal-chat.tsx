"use client"

import type React from "react"

import { useState, useRef, useEffect, useMemo } from "react"
import { encode as encodeCl100k } from "gpt-tokenizer/encoding/cl100k_base"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Slider } from "@/components/ui/slider"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible"
import { MessageSquare, Settings, Paperclip, ImageIcon, Mic, Send, Plus, Sparkles, X, Trash2, ChevronDown, ClipboardPaste, RotateCcw, ThumbsUp, ThumbsDown, BarChart2, Activity } from "lucide-react"
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Switch } from "@/components/ui/switch"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
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
    thinking?: string[]
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
  const [activeView, setActiveView] = useState<"chat" | "settings" | "dashboard">("chat")

  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: "1",
      title: "当前对话",
      messages: [
        {
          id: "1",
          role: "assistant",
          content: "您好！请问有什么可以帮助您的？",
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
  const [feedbacks, setFeedbacks] = useState<Record<string, "up" | "down" | undefined>>({})
  const [growthSources, setGrowthSources] = useState({ auto: 0, user: 0, manual: 0 })
  const [effBaseline, setEffBaseline] = useState<null | { avgRetrieval: number; avgGeneration: number; timestamp: string }>(null)
  const [activeSlice, setActiveSlice] = useState<number | null>(null)
  const [growthEvents, setGrowthEvents] = useState<Array<{ type: "auto" | "user" | "manual"; ts: number }>>([])

  // 模型设置状态
  const [streamEnabled, setStreamEnabled] = useState(true)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [maxTokensEnabled, setMaxTokensEnabled] = useState(false)
  const [selectedModel, setSelectedModel] = useState("GPT-4 Turbo")
  const [temperature, setTemperature] = useState(0.7)
  const [chainLength, setChainLength] = useState(2)
  const [chainLevel, setChainLevel] = useState("basic")

  const fileInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [messages])

  useEffect(() => {
    try {
      const raw = localStorage.getItem("effBaseline")
      if (raw) {
        const parsed = JSON.parse(raw)
        if (parsed && typeof parsed.avgRetrieval === "number" && typeof parsed.avgGeneration === "number") {
          setEffBaseline(parsed)
        }
      }
    } catch {}
  }, [])

  useEffect(() => {
    try {
      const raw = localStorage.getItem("growthEvents")
      if (raw) {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed)) setGrowthEvents(parsed)
      }
    } catch {}
  }, [])

  const isDefaultGreeting = (text: string) => text.trim() === "您好！请问有什么可以帮助您的？"

  const handleFeedback = (messageId: string, rating: "up" | "down") => {
    setFeedbacks((prev) => ({ ...prev, [messageId]: rating }))
  }

  const analytics = useMemo(() => {
    const all = conversations.flatMap((c) => c.messages)
    const assistants = all.filter((m) => m.role === "assistant" && !isDefaultGreeting(m.content))
    const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0)
    const retrievals = assistants.map((m) => m.metadata?.retrievalTime || 0).filter((v) => v > 0)
    const generations = assistants.map((m) => m.metadata?.generationTime || 0).filter((v) => v > 0)
    const completionTokens = assistants.map((m) => encodeCl100k(m.content).length).filter((v) => v > 0)
    const up = Object.entries(feedbacks).filter(([, v]) => v === "up").length
    const down = Object.entries(feedbacks).filter(([, v]) => v === "down").length
    const sourceMap: Record<string, number> = {}
    assistants.forEach((m) => {
      const src = m.metadata?.sources || []
      src.forEach((s) => {
        sourceMap[s] = (sourceMap[s] || 0) + 1
      })
    })
    const sources = Object.entries(sourceMap)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
    return {
      totalMessages: all.length,
      assistantMessages: assistants.length,
      avgRetrieval: avg(retrievals),
      avgGeneration: avg(generations),
      avgCompletionTokens: avg(completionTokens),
      up,
      down,
      sources,
    }
  }, [conversations, feedbacks])

  const copyAssistantToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {}
  }

  const saveEfficiencyBaseline = () => {
    const b = { avgRetrieval: analytics.avgRetrieval, avgGeneration: analytics.avgGeneration, timestamp: new Date().toISOString() }
    setEffBaseline(b)
    try {
      localStorage.setItem("effBaseline", JSON.stringify(b))
    } catch {}
  }

  const pushGrowthEvent = (type: "auto" | "user" | "manual", count = 1) => {
    const now = Date.now()
    const events = Array.from({ length: count }, () => ({ type, ts: now }))
    setGrowthEvents((prev) => {
      const next = [...prev, ...events]
      try {
        localStorage.setItem("growthEvents", JSON.stringify(next))
      } catch {}
      return next
    })
  }

  const weeklyCounts = useMemo(() => {
    const weekAgo = Date.now() - 7 * 24 * 3600 * 1000
    const counts = { auto: 0, user: 0, manual: 0 }
    growthEvents.forEach((e) => {
      if (e.ts >= weekAgo) counts[e.type] += 1
    })
    return counts
  }, [growthEvents])

  const lighten = (hex: string, ratio: number) => {
    const h = hex.replace("#", "")
    const bigint = parseInt(h, 16)
    const r = (bigint >> 16) & 255
    const g = (bigint >> 8) & 255
    const b = bigint & 255
    const nr = Math.round(r * (1 - ratio) + 255 * ratio)
    const ng = Math.round(g * (1 - ratio) + 255 * ratio)
    const nb = Math.round(b * (1 - ratio) + 255 * ratio)
    const toHex = (v: number) => v.toString(16).padStart(2, "0")
    return `#${toHex(nr)}${toHex(ng)}${toHex(nb)}`
  }

  const growthData = useMemo(
    () => [
      { key: "auto" as const, name: "自动学习", value: growthSources.auto, color: "#547CAE" },
      { key: "user" as const, name: "用户交互", value: growthSources.user, color: "#82AADB" },
      { key: "manual" as const, name: "人工补充", value: growthSources.manual, color: "#CAD8D9" },
    ],
    [growthSources],
  )
  const totalGrowth = useMemo(() => growthData.reduce((s, d) => s + d.value, 0), [growthData])

  const GrowthTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const p = payload[0]
      const entry = p.payload
      const percent = totalGrowth > 0 ? Math.round((entry.value / totalGrowth) * 100) : 0
      const w = weeklyCounts[entry.key as "auto" | "user" | "manual"] || 0
      return (
        <div className="bg-popover text-popover-foreground border border-border rounded-lg p-2 text-xs">
          {`${entry.name}占比 ${percent}%，本周新增 ${w} 条`}
        </div>
      )
    }
    return null
  }

  const regenerateAnswer = async (assistantMessageId: string) => {
    const conv = conversations.find((c) => c.id === currentConversationId)
    if (!conv) return
    const idx = conv.messages.findIndex((m) => m.id === assistantMessageId)
    let userMsg: Message | undefined
    for (let i = idx - 1; i >= 0; i--) {
      if (conv.messages[i].role === "user") {
        userMsg = conv.messages[i]
        break
      }
    }
    if (!userMsg) {
      const lastUser = [...conv.messages].reverse().find((m) => m.role === "user")
      if (!lastUser) return
      userMsg = lastUser
    }

    setIsLoading(true)
    try {
      const questionResponse = await fetch("/api/chat/question", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg.content,
          conversationId: currentConversationId,
          attachments: (userMsg.attachments || []).map((a) => ({ type: a.type, name: a.name })),
          stream: streamEnabled,
          maxTokens: maxTokensEnabled ? maxTokens : undefined,
          chainLength,
        }),
      })
      const questionData = await questionResponse.json()

      const retrieveResponse = await fetch("/api/chat/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questionId: questionData.data.questionId,
          query: userMsg.content,
        }),
      })
      const retrieveData = await retrieveResponse.json()

      const generateResponse = await fetch("/api/chat/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questionId: questionData.data.questionId,
          query: userMsg.content,
          retrievedDocs: retrieveData.data.results,
          conversationHistory: conv.messages.slice(-5),
          chainLength,
        }),
      })
      const generateData = await generateResponse.json()

      const meta = {
        questionId: questionData.data.questionId,
        retrievalTime: retrieveData.data.retrievalTime,
        generationTime: generateData.data.generationTime,
        confidence: generateData.data.confidence,
        sources: generateData.data.sources,
        thinking: generateData.data.thinking,
      }

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentConversationId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: streamEnabled ? "" : generateData.data.answer, metadata: meta }
                    : m,
                ),
                updatedAt: new Date(),
              }
            : c,
        ),
      )

      setGrowthSources((s) => ({ ...s, auto: s.auto + 1 }))
      pushGrowthEvent("auto", 1)

      if (streamEnabled) {
        const full = generateData.data.answer as string
        let i = 0
        const step = 8
        if (streamIntervalRef.current) clearInterval(streamIntervalRef.current)
        streamIntervalRef.current = setInterval(() => {
          i += step
          const slice = full.slice(0, i)
          setConversations((prev) =>
            prev.map((c) =>
              c.id === currentConversationId
                ? {
                    ...c,
                    messages: c.messages.map((m) => (m.id === assistantMessageId ? { ...m, content: slice } : m)),
                    updatedAt: new Date(),
                  }
                : c,
            ),
          )
          if (i >= full.length) {
            if (streamIntervalRef.current) {
              clearInterval(streamIntervalRef.current)
              streamIntervalRef.current = null
            }
          }
        }, 30)
      }
    } catch (e) {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentConversationId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMessageId ? { ...m, content: "抱歉，重新生成失败，请稍后重试。" } : m,
                ),
                updatedAt: new Date(),
              }
            : c,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  const defaultMaxTokens = useMemo(() => {
    if (selectedModel === "GPT-4 Turbo") return 4096
    return 4096
  }, [selectedModel])

  const inputTokens = useMemo(() => {
    if (!inputValue) return 0
    if (selectedModel === "GPT-4 Turbo") return encodeCl100k(inputValue).length
    return encodeCl100k(inputValue).length
  }, [inputValue, selectedModel])

  const maxTokensDisplay = maxTokensEnabled ? maxTokens : defaultMaxTokens

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

    const attCountForGrowth = attachments.length
    setGrowthSources((s) => ({ ...s, user: s.user + 1, manual: s.manual + attCountForGrowth }))
    pushGrowthEvent("user", 1)
    if (attCountForGrowth > 0) pushGrowthEvent("manual", attCountForGrowth)

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
          chainLength,
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
          chainLength,
        }),
      })
      const generateData = await generateResponse.json()
      console.log("[v0] 答案生成响应:", generateData)

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: streamEnabled ? "" : generateData.data.answer,
        metadata: {
          questionId: questionData.data.questionId,
          retrievalTime: retrieveData.data.retrievalTime,
          generationTime: generateData.data.generationTime,
          confidence: generateData.data.confidence,
          sources: generateData.data.sources,
          thinking: generateData.data.thinking,
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

      setGrowthSources((s) => ({ ...s, auto: s.auto + 1 }))
      pushGrowthEvent("auto", 1)

      if (streamEnabled) {
        const full = generateData.data.answer as string
        let i = 0
        const step = 8
        if (streamIntervalRef.current) {
          clearInterval(streamIntervalRef.current)
        }
        streamIntervalRef.current = setInterval(() => {
          i += step
          const slice = full.slice(0, i)
          setConversations((prev) =>
            prev.map((conv) =>
              conv.id === currentConversationId
                ? {
                    ...conv,
                    messages: conv.messages.map((m) =>
                      m.id === assistantMessage.id ? { ...m, content: slice } : m,
                    ),
                    updatedAt: new Date(),
                  }
                : conv,
            ),
          )
          if (i >= full.length) {
            if (streamIntervalRef.current) {
              clearInterval(streamIntervalRef.current)
              streamIntervalRef.current = null
            }
          }
        }, 30)
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
          content: "您好！请问有什么可以帮助您的？",
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

  return (
    <div className="flex h-screen max-h-screen bg-background overflow-hidden">
      {/* 左侧导航栏 */}
      <aside className="w-72 border-r border-border bg-sidebar flex flex-col shadow-sm">
        <div className="p-5 border-b border-sidebar-border">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-md">
              <Sparkles className="w-5 h-5 text-primary-foreground" />
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

        <ScrollArea className="flex-1 min-h-0 p-4">
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
          <button
            onClick={() => setActiveView("dashboard")}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-sidebar-foreground",
              activeView === "dashboard" ? "bg-primary text-primary-foreground shadow-md" : "hover:bg-sidebar-accent",
            )}
          >
            <BarChart2 className="w-4 h-4" />
            <span className="text-sm font-medium">看板</span>
          </button>
        </div>
      </aside>

      {/* 主内容区域 */}
      <main className="flex-1 flex flex-col min-w-0 min-h-0">
        {activeView === "chat" ? (
          <>
            {/* 聊天消息区域 */}
            <ScrollArea className="flex-1 h-full min-h-0 overflow-y-auto px-6 py-8">
              <div className="max-w-4xl mx-auto space-y-8 pb-28">
                {messages.map((message) => (
                  message.role === "user" ? (
                    <div
                      key={message.id}
                      className="flex items-center gap-3 animate-in fade-in slide-in-from-bottom-4 duration-500"
                    >
                      <Avatar className="w-8 h-8 border-2 border-secondary shadow-sm">
                        <AvatarFallback className="bg-secondary text-secondary-foreground text-xs font-medium">
                          我
                        </AvatarFallback>
                      </Avatar>
                      <div className="space-y-2 max-w-[75%]">
                        {message.attachments && message.attachments.length > 0 && (
                          <div className="space-y-2">
                            {message.attachments.map((attachment, idx) => (
                              <div key={idx}>
                                {attachment.type === "image" ? (
                                  <div className="rounded-lg overflow-hidden border border-border/50">
                                    <img
                                      src={attachment.url || "/placeholder.svg"}
                                      alt={attachment.name}
                                      className="max-w-full h-auto max-h-64 object-cover"
                                    />
                                    <div className="flex items-center gap-2 p-2 text-xs bg-muted/50">
                                      <ImageIcon className="w-3.5 h-3.5" />
                                      <span className="truncate">{attachment.name}</span>
                                    </div>
                                  </div>
                                ) : attachment.type === "audio" ? (
                                  <div className="rounded-lg overflow-hidden border border-border/50 bg-muted/50">
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
                                  <div className="flex items-center gap-2 p-2.5 rounded-lg bg-muted/50">
                                    <Paperclip className="w-4 h-4" />
                                    <span className="text-sm truncate">{attachment.name}</span>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                        <div className="text-base leading-relaxed whitespace-pre-wrap">
                          {message.content}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div
                      key={message.id}
                      className="animate-in fade-in slide-in-from-bottom-4 duration-500"
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
                                  <div className="flex items-center gap-2 p-2 text-xs bg-muted/50">
                                    <ImageIcon className="w-3.5 h-3.5" />
                                    <span className="truncate">{attachment.name}</span>
                                  </div>
                                </div>
                              ) : attachment.type === "audio" ? (
                                <div className="rounded-lg overflow-hidden border border-border/50 bg-muted/50">
                                  <audio controls className="w-full h-10">
                                    <source src={attachment.url} type="audio/webm" />
                                    您的浏览器不支持音频播放
                                  </audio>
                                  <div className="flex items中心 gap-2 p-2 text-xs border-t border-border/30">
                                    <Mic className="w-3.5 h-3.5" />
                                    <span className="truncate">{attachment.name}</span>
                                  </div>
                                </div>
                              ) : (
                                <div className="flex items-center gap-2 p-2.5 rounded-lg bg-muted/50">
                                  <Paperclip className="w-4 h-4" />
                                  <span className="text-sm truncate">{attachment.name}</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="text-base leading-relaxed whitespace-pre-wrap">
                        {message.content}
                      </div>
                      {message.metadata && message.role === "assistant" && (
                        <Collapsible>
                          <div className="mt-3 flex items-center gap-2">
                            <CollapsibleTrigger className="text-xs text-muted-foreground hover:text-foreground flex items-center">
                              <ChevronDown className="w-4 h-4 transition-transform data-[state=open]:rotate-180" />
                            </CollapsibleTrigger>
                            <button onClick={() => copyAssistantToClipboard(message.content)} className="text-muted-foreground hover:text-foreground" aria-label="粘贴" title="粘贴">
                              <ClipboardPaste className="w-4 h-4" />
                            </button>
                            <button onClick={() => regenerateAnswer(message.id)} className="text-muted-foreground hover:text-foreground" aria-label="重新生成" title="重新生成">
                              <RotateCcw className="w-4 h-4" />
                            </button>
                            <button onClick={() => handleFeedback(message.id, "up")} className={cn("text-muted-foreground hover:text-foreground", feedbacks[message.id] === "up" && "text-primary")} aria-label="赞" title="赞">
                              <ThumbsUp className="w-4 h-4" />
                            </button>
                            <button onClick={() => handleFeedback(message.id, "down")} className={cn("text-muted-foreground hover:text-foreground", feedbacks[message.id] === "down" && "text-destructive")} aria-label="踩" title="踩">
                              <ThumbsDown className="w-4 h-4" />
                            </button>
                          </div>
                          <CollapsibleContent>
                            <div className="pt-3 border-t border-border/30 text-xs text-muted-foreground space-y-1">
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
                              {message.metadata.thinking && message.metadata.thinking.length > 0 && (
                                <div className="space-y-1">
                                  <div>思考过程:</div>
                                  {message.metadata.thinking.map((t, i) => (
                                    <div key={i}>{t}</div>
                                  ))}
                                </div>
                              )}
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      )}
                      {message.role === "assistant" && !isDefaultGreeting(message.content) && (
                        <div className="mt-2 text-[11px] text-muted-foreground">
                          tokens：
                          {selectedModel === "GPT-4 Turbo"
                            ? encodeCl100k(message.content).length
                            : encodeCl100k(message.content).length}
                          /{maxTokensDisplay}
                        </div>
                      )}
                    </div>
                  )
                ))}
                {isLoading && (
                  <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="rounded-2xl px-5 py-3.5 bg-card border border-border">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-primary animate-bounce" />
                        <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.2s]" />
                        <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.4s]" />
                        <span className="text-sm text-muted-foreground ml-2">AI正在思考...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} className="mb-16" />
              </div>
            </ScrollArea>

            {/* 输入区域 */}
            <div className="border-t border-border bg-card/50 backdrop-blur-sm sticky bottom-0 z-10">
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
                      ref={chatInputRef}
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault()
                          handleSend()
                        }
                      }}
                      placeholder="输入消息或添加文件、图片、录音..."
                      className="pr-36 min-h-[52px] resize-none bg-background border-border shadow-sm rounded-xl text-sm"
                      disabled={isRecording || isLoading}
                    />
                    <div className="absolute right-2 bottom-2 flex items-center gap-1">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-9 w-9 hover:bg-accent"
                        onClick={() => handleFileSelect("file")}
                        disabled={isRecording || isLoading}
                      >
                        <Paperclip className="w-4 h-4" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-9 w-9 hover:bg-accent"
                        onClick={() => handleFileSelect("image")}
                        disabled={isRecording || isLoading}
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
                        disabled={isLoading}
                      >
                        <Mic className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  <Button
                    onClick={handleSend}
                    size="icon"
                    className="h-[52px] w-[52px] flex-shrink-0 rounded-xl shadow-md hover:shadow-lg transition-shadow"
                    disabled={isRecording || isLoading}
                  >
                    <Send className="w-5 h-5" />
                  </Button>
                </div>
                <div className="mt-3 flex justify-end">
                  <p className="text-xs text-muted-foreground">tokens：{inputTokens}/{maxTokensDisplay}</p>
                </div>
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
        ) : activeView === "settings" ? (
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
                      <label className="text-sm font-medium text-foreground mb-2 block">思维链长度</label>
                      <RadioGroup
                        value={chainLevel}
                        onValueChange={(v) => {
                          setChainLevel(v)
                          setChainLength(v === "basic" ? 2 : v === "medium" ? 6 : 10)
                        }}
                        className="grid grid-cols-3 gap-2"
                      >
                        <label className="flex items-center gap-2 rounded-xl border border-border px-3 py-2 cursor-pointer">
                          <RadioGroupItem value="basic" />
                          <span className="text-sm">初级</span>
                        </label>
                        <label className="flex items-center gap-2 rounded-xl border border-border px-3 py-2 cursor-pointer">
                          <RadioGroupItem value="medium" />
                          <span className="text-sm">中级</span>
                        </label>
                        <label className="flex items-center gap-2 rounded-xl border border-border px-3 py-2 cursor-pointer">
                          <RadioGroupItem value="advanced" />
                          <span className="text-sm">高级</span>
                        </label>
                      </RadioGroup>
                      <div className="text-xs text-muted-foreground mt-2">
                        {chainLevel === "basic" && "0–2步"}
                        {chainLevel === "medium" && "4–6步"}
                        {chainLevel === "advanced" && "10+步"}
                      </div>
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-medium text-foreground">温度</label>
                        <span className="text-xs text-muted-foreground">{temperature.toFixed(1)}</span>
                      </div>
                      <Slider
                        min={0}
                        max={1}
                        step={0.1}
                        value={[temperature]}
                        onValueChange={(vals) => setTemperature(vals[0])}
                        className="w-full"
                      />
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
        ) : (
          <div className="flex-1 px-6 py-8 overflow-y-auto">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-3xl font-semibold mb-8 text-balance">看板</h2>
              <div className="mt-6 bg-card border border-border rounded-2xl p-6">
                <div className="text-sm font-medium mb-3">知识成长来源占比</div>
                {totalGrowth === 0 ? (
                  <div className="text-xs text-muted-foreground">暂无数据</div>
                ) : (
                  <div className="w-full h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <defs>
                          {growthData.map((d) => (
                            <linearGradient id={`grad-${d.key}`} key={d.key} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={lighten(d.color, 0.1)} />
                              <stop offset="100%" stopColor={d.color} />
                            </linearGradient>
                          ))}
                        </defs>
                        <Pie
                          dataKey="value"
                          data={growthData}
                          outerRadius={90}
                          label
                          onMouseEnter={(_, idx) => setActiveSlice(idx)}
                          onMouseLeave={() => setActiveSlice(null)}
                        >
                          {growthData.map((d, idx) => (
                            <Cell key={d.key} fill={activeSlice === idx ? `url(#grad-${d.key})` : d.color} />
                          ))}
                        </Pie>
                        <Tooltip content={<GrowthTooltip />} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
              <div className="mt-6 bg-card border border-border rounded-2xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-sm font-medium">交互效率</div>
                  <Button onClick={saveEfficiencyBaseline} size="sm" className="rounded-xl">设为对比基线</Button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="bg-muted rounded-xl p-4">
                    <div className="text-xs text-muted-foreground">当前平均检索耗时</div>
                    <div className="text-xl font-semibold mt-1">{analytics.avgRetrieval.toFixed(2)}s</div>
                  </div>
                  <div className="bg-muted rounded-xl p-4">
                    <div className="text-xs text-muted-foreground">当前平均生成耗时</div>
                    <div className="text-xl font-semibold mt-1">{analytics.avgGeneration.toFixed(2)}s</div>
                  </div>
                  <div className="bg-muted rounded-xl p-4">
                    <div className="text-xs text-muted-foreground">基线时间</div>
                    <div className="text-sm mt-1">{effBaseline ? new Date(effBaseline.timestamp).toLocaleString("zh-CN") : "未设置"}</div>
                  </div>
                </div>
                {effBaseline && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-muted rounded-xl p-4">
                      <div className="text-xs text-muted-foreground">检索耗时提升对比</div>
                      <div className="text-xl font-semibold mt-1">
                        {effBaseline.avgRetrieval > 0
                          ? `${(((effBaseline.avgRetrieval - analytics.avgRetrieval) / effBaseline.avgRetrieval) * 100).toFixed(0)}%`
                          : "-"}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">相对上一版本</div>
                    </div>
                    <div className="bg-muted rounded-xl p-4">
                      <div className="text-xs text-muted-foreground">生成耗时提升对比</div>
                      <div className="text-xl font-semibold mt-1">
                        {effBaseline.avgGeneration > 0
                          ? `${(((effBaseline.avgGeneration - analytics.avgGeneration) / effBaseline.avgGeneration) * 100).toFixed(0)}%`
                          : "-"}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">相对上一版本</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
