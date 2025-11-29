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
import { MessageSquare, Settings, Paperclip, ImageIcon, Mic, Send, Plus, Sparkles, X, RotateCcw, ThumbsUp, ThumbsDown, BarChart2, Activity, User, Home, Hash, Globe, Hammer, ChevronRight } from "lucide-react"
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts"
import { HistoryItem } from "@/components/ui/history-item"
import { Switch } from "@/components/ui/switch"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { cn } from "@/lib/utils"
import { HaloButton } from "@/components/ui/halo-button"

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
  const [sidebarTab, setSidebarTab] = useState<"topics" | "settings" | "dashboard">("topics")

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

  const [selectedModel, setSelectedModel] = useState("Qwen3-8B")
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [streamEnabled, setStreamEnabled] = useState(true)
  const [maxTokensEnabled, setMaxTokensEnabled] = useState(false)
  const [chainLevel, setChainLevel] = useState("medium") // basic, medium, advanced
  const [chainLength, setChainLength] = useState(6)

  const [growthEvents, setGrowthEvents] = useState<Array<{ type: "auto" | "user" | "manual"; ts: number }>>([])

  const scrollRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isLoading]) //Added isLoading to auto scroll when generating

  useEffect(() => {
    try {
      const saved = localStorage.getItem("effBaseline")
      if (saved) setEffBaseline(JSON.parse(saved))
      const savedEvents = localStorage.getItem("growthEvents")
      if (savedEvents) {
        const parsed = JSON.parse(savedEvents)
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

  const efficiencySeries = useMemo(() => {
    const all = conversations.flatMap((c) => c.messages)
    const assistants = all.filter((m) => m.role === "assistant" && !isDefaultGreeting(m.content))
    const baseR = effBaseline?.avgRetrieval || 0
    const baseG = effBaseline?.avgGeneration || 0
    return assistants
      .filter((m) => typeof m.metadata?.retrievalTime === "number" || typeof m.metadata?.generationTime === "number")
      .map((m, i) => {
        const rt = m.metadata?.retrievalTime || 0
        const gt = m.metadata?.generationTime || 0
        const rGain = baseR > 0 ? ((baseR - rt) / baseR) * 100 : null
        const gGain = baseG > 0 ? ((baseG - gt) / baseG) * 100 : null
        return {
          ts: Number(m.id) || Date.now(),
          index: i + 1,
          retrievalGain: rGain,
          generationGain: gGain,
          retrievalTime: rt,
          generationTime: gt,
        }
      })
  }, [conversations, effBaseline])

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
    <div className="flex h-screen w-full bg-[#FDFDFD] text-zinc-800 font-sans selection:bg-primary/20">
      {/* Sidebar */}
      <aside className="w-[260px] flex-shrink-0 flex flex-col bg-[#FAFAFA] border-r border-zinc-100">
        {/* Home / Search Bar */}
        <div className="h-14 flex items-center px-4 border-b border-zinc-50/50">
          <div className="flex-1 flex items-center gap-2 bg-white border border-zinc-200/60 rounded-xl px-3 py-1.5 shadow-sm hover:shadow transition-all cursor-text group">
            <Home className="w-4 h-4 text-zinc-400 group-hover:text-zinc-600 transition-colors" />
            <span className="text-sm text-zinc-500 font-medium">首页</span>
          </div>
          <Button variant="ghost" size="icon" onClick={createNewConversation} className="ml-2 h-8 w-8 rounded-full hover:bg-zinc-200/50 text-zinc-500">
            <Plus className="w-5 h-5" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex items-center px-4 py-2 gap-6 border-b border-zinc-100/50">
          <button
            onClick={() => {
              setSidebarTab("topics")
              setActiveView("chat")
            }}
            className={cn(
              "text-sm font-medium pb-2 border-b-2 transition-all",
              sidebarTab === "topics" ? "text-zinc-900 border-[#00B894]" : "text-zinc-400 border-transparent hover:text-zinc-600"
            )}
          >
            话题
          </button>
          <button
            onClick={() => {
               setSidebarTab("dashboard")
               setActiveView("dashboard")
            }}
            className={cn(
              "text-sm font-medium pb-2 border-b-2 transition-all",
              sidebarTab === "dashboard" ? "text-zinc-900 border-[#00B894]" : "text-zinc-400 border-transparent hover:text-zinc-600"
            )}
          >
            看板
          </button>
          <button
            onClick={() => {
               setSidebarTab("settings")
               setActiveView("settings")
            }}
            className={cn(
              "text-sm font-medium pb-2 border-b-2 transition-all",
              sidebarTab === "settings" ? "text-zinc-900 border-[#00B894]" : "text-zinc-400 border-transparent hover:text-zinc-600"
            )}
          >
            设置
          </button>
        </div>

        {/* Sidebar Content */}
        <ScrollArea className="flex-1 px-3 py-4">

          
          {sidebarTab === "topics" && (
             <div className="space-y-2">
               <div className="mb-4 px-1">
                 <HaloButton onClick={createNewConversation} icon={<Plus className="w-4 h-4" />}>
                   新建对话
                 </HaloButton>
               </div>
               {conversations
                  .slice()
                  .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
                  .map((c) => (
                 <HistoryItem
                    key={c.id}
                    title={c.title}
                    isActive={currentConversationId === c.id}
                    onClick={() => {
                      switchConversation(c.id)
                      setActiveView("chat")
                    }}
                    onDelete={(e) => deleteConversation(c.id, e)}
                    preview={c.messages[c.messages.length - 1]?.content || "暂无消息"}
                    date={c.updatedAt.toLocaleString("zh-CN", { month: "numeric", day: "numeric" })}
                 />
               ))}
             </div>
          )}
        </ScrollArea>
        

      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 bg-white relative overflow-hidden">
        {/* Header */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-zinc-50 bg-white/80 backdrop-blur-sm z-10">

           <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" className="h-8 w-8 text-zinc-400 hover:text-zinc-600" onClick={() => {
                setActiveView("dashboard")
                setSidebarTab("dashboard")
              }}><Activity className="w-4 h-4" /></Button>
              <Button variant="ghost" size="icon" className="h-8 w-8 text-zinc-400 hover:text-zinc-600" onClick={() => {
                setActiveView("settings")
                setSidebarTab("settings")
              }}><Settings className="w-4 h-4" /></Button>
           </div>
        </header>

        {activeView === "chat" ? (
          <>
            <ScrollArea className="flex-1 px-8 py-6" ref={scrollRef}>
               <div className="max-w-3xl mx-auto space-y-10 pb-32">
                  {messages.map((message) => (
                    <div key={message.id} className="group">
                       {message.role === "user" ? (
                         <div className="flex justify-end">
                           <div className="bg-zinc-100 text-zinc-800 px-5 py-3 rounded-2xl rounded-tr-sm max-w-[80%] text-base leading-7 shadow-sm">
                             {message.content}
                           </div>
                         </div>
                       ) : (
                         <div className="flex gap-4">
                            <div className="flex-1 space-y-4 overflow-hidden">
                              {/* Message Content */}
                              <div className="prose prose-zinc max-w-none text-zinc-700 leading-7">
                                {message.content.split('\n').map((line, i) => (
                                  <p key={i} className="mb-2">{line}</p>
                                ))}
                              </div>

                              {/* Attachments Display */}
                              {message.attachments && message.attachments.length > 0 && (
                                <div className="flex flex-wrap gap-2">
                                  {message.attachments.map((att, idx) => (
                                     <div key={idx} className="flex items-center gap-2 bg-zinc-50 border border-zinc-100 rounded-lg px-3 py-2 text-sm text-zinc-600">
                                        {att.type === 'image' ? <ImageIcon className="w-4 h-4" /> : att.type === 'audio' ? <Mic className="w-4 h-4" /> : <Paperclip className="w-4 h-4" />}
                                        <span className="truncate max-w-[200px]">{att.name}</span>
                                     </div>
                                  ))}
                                </div>
                              )}
                              
                              {/* Metadata / Example Card */}
                              {message.metadata && (
                                 <div className="flex flex-col gap-2 mt-2">
                                    <div className="flex flex-wrap gap-2">
                                        {message.metadata.thinking && message.metadata.thinking.length > 0 && (
                                            <div className="bg-blue-50 text-blue-700 text-xs px-2 py-1 rounded-md">
                                                思考中...
                                            </div>
                                        )}
                                        {message.metadata.retrievalTime && (
                                            <div className="text-xs text-zinc-400">检索: {message.metadata.retrievalTime.toFixed(2)}s</div>
                                        )}
                                        {message.metadata.generationTime && (
                                            <div className="text-xs text-zinc-400">生成: {message.metadata.generationTime.toFixed(2)}s</div>
                                        )}
                                    </div>
                                    {/* If there are sources, show them */}
                                    {message.metadata.sources && message.metadata.sources.length > 0 && (
                                        <div className="text-xs text-zinc-500 bg-zinc-50 p-2 rounded-lg border border-zinc-100">
                                            来源: {message.metadata.sources.join(", ")}
                                        </div>
                                    )}
                                 </div>
                              )}
                            </div>
                         </div>
                       )}
                    </div>
                  ))}
               </div>
            </ScrollArea>

            {/* Input Area - Floating Bottom */}
            <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-white via-white to-transparent z-20">
              <div className="max-w-3xl mx-auto">
                <div className="bg-white rounded-[2rem] border border-zinc-200 shadow-[0_8px_40px_-12px_rgba(0,0,0,0.1)] overflow-hidden transition-all focus-within:ring-1 focus-within:ring-primary/20 focus-within:border-primary/50">


                  {/* Textarea */}
                  <div className="px-4 py-2">
                    <Input
                       className="border-0 shadow-none focus-visible:ring-0 p-0 text-base min-h-[40px] resize-none bg-transparent placeholder:text-zinc-300"
                       placeholder="在这里输入消息，按 Enter 发送..."
                       value={inputValue}
                       onChange={(e) => setInputValue(e.target.value)}
                       onKeyDown={(e) => {
                         if (e.key === "Enter" && !e.shiftKey) {
                           e.preventDefault()
                           handleSend()
                         }
                       }}
                    />
                  </div>

                  {/* Bottom Toolbar */}
                  <div className="flex items-center justify-between px-3 py-2 bg-white">
                     <div className="flex items-center gap-1">
                        <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded-full" onClick={() => handleFileSelect('file')}><Plus className="w-5 h-5" /></Button>
                        <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded-full" onClick={() => handleFileSelect('image')}><ImageIcon className="w-5 h-5" /></Button>
                        <Button variant="ghost" size="icon" className={cn("h-9 w-9 rounded-full", isRecording ? "text-red-500 bg-red-50" : "text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100")} onClick={toggleRecording}><Mic className="w-5 h-5" /></Button>
                        <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded-full"><Globe className="w-5 h-5" /></Button>
                        <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded-full"><Hammer className="w-5 h-5" /></Button>
                     </div>
                     <div className="flex items-center gap-4">
                        <div className="text-[10px] text-zinc-300 font-mono">= {chainLength}/10 ↑ {conversations.length}</div>
                        <Button 
                          onClick={handleSend} 
                          disabled={isLoading || (!inputValue.trim() && attachments.length === 0)}
                          className={cn("rounded-full w-9 h-9 p-0 flex items-center justify-center transition-all", (inputValue.trim() || attachments.length > 0) ? "bg-black text-white shadow-md hover:bg-zinc-800" : "bg-zinc-100 text-zinc-300")}
                        >
                           {isLoading ? <RotateCcw className="w-5 h-5 animate-spin" /> : <ChevronRight className="w-5 h-5" />}
                        </Button>
                     </div>
                  </div>
                </div>
                
                {/* Attachments Preview */}
                {attachments.length > 0 && (
                  <div className="mt-2 flex gap-2 overflow-x-auto pb-2">
                    {attachments.map((att, i) => (
                      <div key={i} className="relative bg-white border border-zinc-200 rounded-lg p-2 flex items-center gap-2 shadow-sm min-w-[120px]">
                         <span className="text-xs truncate max-w-[100px]">{att.name}</span>
                         <button onClick={() => removeAttachment(i)} className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full p-0.5"><X className="w-3 h-3" /></button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            {/* Hidden inputs */}
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={(e) => handleFileChange(e, "file")}
            />
            <input
              type="file"
              ref={imageInputRef}
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFileChange(e, "image")}
            />
          </>
        ) : activeView === "settings" ? (
          <div className="flex-1 px-6 py-8 overflow-y-auto">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-3xl font-semibold mb-8 text-balance text-zinc-800">设置</h2>
              <div className="space-y-6">
                <div className="bg-white border border-zinc-200 rounded-2xl p-6 shadow-sm">
                  <h3 className="font-semibold text-lg mb-5 flex items-center gap-2 text-zinc-800">
                    <div className="w-2 h-2 rounded-full bg-primary"></div>
                    模型设置
                  </h3>
                  <div className="space-y-5">
                    <div>
                      <label className="text-sm font-medium text-zinc-700 mb-2 block">思维链长度</label>
                      <RadioGroup
                        value={chainLevel}
                        onValueChange={(v) => {
                          setChainLevel(v)
                          setChainLength(v === "basic" ? 2 : v === "medium" ? 6 : 10)
                        }}
                        className="grid grid-cols-3 gap-2"
                      >
                        <label className="flex items-center gap-2 rounded-xl border border-zinc-200 px-3 py-2 cursor-pointer hover:bg-zinc-50 transition-colors [&:has([data-state=checked])]:border-primary [&:has([data-state=checked])]:bg-primary/5">
                          <RadioGroupItem value="basic" />
                          <span className="text-sm">初级</span>
                        </label>
                        <label className="flex items-center gap-2 rounded-xl border border-zinc-200 px-3 py-2 cursor-pointer hover:bg-zinc-50 transition-colors [&:has([data-state=checked])]:border-primary [&:has([data-state=checked])]:bg-primary/5">
                          <RadioGroupItem value="medium" />
                          <span className="text-sm">中级</span>
                        </label>
                        <label className="flex items-center gap-2 rounded-xl border border-zinc-200 px-3 py-2 cursor-pointer hover:bg-zinc-50 transition-colors [&:has([data-state=checked])]:border-primary [&:has([data-state=checked])]:bg-primary/5">
                          <RadioGroupItem value="advanced" />
                          <span className="text-sm">高级</span>
                        </label>
                      </RadioGroup>
                      <div className="text-xs text-zinc-400 mt-2">
                        {chainLevel === "basic" && "0–2步"}
                        {chainLevel === "medium" && "4–6步"}
                        {chainLevel === "advanced" && "10+步"}
                      </div>
                    </div>
                    {/* More settings can be added here if needed, keeping it simple for now */}
                    <div>
                       <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-medium text-zinc-700">温度</label>
                        <span className="text-xs text-zinc-500">{temperature.toFixed(1)}</span>
                      </div>
                      <Slider
                        value={[temperature]}
                        min={0}
                        max={1}
                        step={0.1}
                        onValueChange={([v]) => setTemperature(v)}
                        className="py-2"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          // Dashboard View
          <div className="flex-1 px-6 py-8 overflow-y-auto bg-zinc-50/50">
             <div className="max-w-5xl mx-auto">
               <div className="flex items-center justify-between mb-8">
                 <h2 className="text-2xl font-semibold text-zinc-800">数据看板</h2>
                 <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={saveEfficiencyBaseline} className="bg-white">保存基准</Button>
                 </div>
               </div>
               
               <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                  <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm">
                     <div className="text-sm text-zinc-500 mb-1">总消息数</div>
                     <div className="text-3xl font-bold text-zinc-900">{analytics.totalMessages}</div>
                  </div>
                  <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm">
                     <div className="text-sm text-zinc-500 mb-1">助手回复</div>
                     <div className="text-3xl font-bold text-zinc-900">{analytics.assistantMessages}</div>
                  </div>
                  <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm">
                     <div className="text-sm text-zinc-500 mb-1">平均生成耗时</div>
                     <div className="text-3xl font-bold text-zinc-900">{analytics.avgGeneration.toFixed(2)}s</div>
                  </div>
               </div>

               <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm mb-6">
                  <h3 className="font-medium text-zinc-800 mb-6">耗时趋势</h3>
                  <div className="h-[300px] w-full">
                     {efficiencySeries.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={efficiencySeries}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                          <XAxis dataKey="index" tick={{ fontSize: 12, fill: '#a1a1aa' }} axisLine={false} tickLine={false} />
                          <YAxis unit="%" tick={{ fontSize: 12, fill: '#a1a1aa' }} axisLine={false} tickLine={false} />
                          <Tooltip 
                            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="retrievalGain" name="检索提升" stroke="#10b981" strokeWidth={3} dot={false} />
                          <Line type="monotone" dataKey="generationGain" name="生成提升" stroke="#3b82f6" strokeWidth={3} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                     ) : (
                       <div className="flex items-center justify-center h-full text-zinc-400 text-sm">暂无足够数据</div>
                     )}
                  </div>
               </div>
             </div>
          </div>
        )}
      </main>
    </div>
  )
}