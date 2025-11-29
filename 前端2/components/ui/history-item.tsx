"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { Trash2, MessageSquare } from "lucide-react"

interface HistoryItemProps {
  isActive?: boolean
  title: string
  preview: string
  date: string | React.ReactNode
  onClick?: () => void
  onDelete?: (e: React.MouseEvent) => void
}

export function HistoryItem({
  isActive,
  title,
  preview,
  date,
  onClick,
  onDelete,
}: HistoryItemProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "relative group w-full cursor-pointer rounded-xl overflow-hidden transition-all duration-300",
        "border border-transparent", // 基础边框占位
        isActive ? "bg-primary/5" : "hover:bg-sidebar-accent/40"
      )}
    >
      {/* 1. 顶部细线 (Top Thin Line) */}
      <div
        className={cn(
          "absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-primary/60 to-transparent opacity-0 transition-all duration-500 group-hover:opacity-100 group-hover:translate-y-0 translate-y-[-100%]",
          isActive && "opacity-100 translate-y-0 via-primary/80"
        )}
      />

      {/* 2. 滑动光晕叠层 (Sliding Halo Overlay) */}
      {/* 使用 after 伪元素模拟扫光效果 */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-500 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent -translate-x-full group-hover:animate-shine" />
      </div>

      {/* 3. 渐变暗角效果 (Gradient Vignette) */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,transparent_50%,var(--sidebar-accent)_120%)] opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      {/* 内容区域 */}
      <div className="relative z-10 px-4 py-3">
        <div className="flex items-start gap-2 mb-1">
          <MessageSquare
            className={cn(
              "w-4 h-4 mt-0.5 flex-shrink-0 transition-colors duration-300",
              isActive ? "text-primary" : "text-muted-foreground group-hover:text-primary/70"
            )}
          />
          <span
            className={cn(
              "text-sm font-medium flex-1 line-clamp-1 transition-colors duration-300",
              isActive ? "text-foreground" : "text-sidebar-foreground group-hover:text-foreground"
            )}
          >
            {title}
          </span>
          
          {onDelete && (
            <button
              onClick={onDelete}
              className="opacity-0 group-hover:opacity-100 transition-all duration-200 p-1 hover:bg-destructive/10 hover:text-destructive text-muted-foreground rounded-md -mr-1"
              title="删除对话"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        <p className={cn(
          "text-xs line-clamp-2 pl-6 transition-colors duration-300",
          isActive ? "text-muted-foreground" : "text-muted-foreground/70 group-hover:text-muted-foreground"
        )}>
          {preview}
        </p>
        
        <div className="mt-1 pl-6 flex items-center justify-between">
          <p className="text-[10px] text-muted-foreground/50 group-hover:text-muted-foreground/70 transition-colors">
            {date}
          </p>
        </div>
      </div>

      {/* 激活状态下的额外微光 */}
      {isActive && (
        <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_20px_rgba(var(--primary),0.05)]" />
      )}
    </div>
  )
}
