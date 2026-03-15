You are given a task to integrate an existing React component in the codebase

The codebase should support:
- shadcn project structure  
- Tailwind CSS
- Typescript

If it doesn't, provide instructions on how to setup project via shadcn CLI, install Tailwind or Typescript.

Determine the default path for components and styles. 
If default path for components is not /components/ui, provide instructions on why it's important to create this folder
Copy-paste this component to /components/ui folder:
```tsx
bouncing-dots.tsx
"use client"

import React from "react"
import { cva } from "class-variance-authority"
import { HTMLMotionProps, motion } from "motion/react"

import { cn } from "@/lib/utils"

const bouncingDotsVariant = cva("flex gap-2 items-center justify-center", {
  variants: {
    messagePlacement: {
      bottom: "flex-col",
      right: "flex-row",
      left: "flex-row-reverse",
    },
  },
  defaultVariants: {
    messagePlacement: "bottom",
  },
})

export interface BouncingDotsProps {
  /**
   * The number of bouncing dots to display.
   * @default 3
   */
  dots?: number
  /**
   * Optional message to display alongside the bouncing dots.
   */
  message?: string
  /**
   * Position of the message relative to the spinner.
   * @default bottom
   */
  messagePlacement?: "bottom" | "left" | "right"
}

export function BouncingDots({
  dots = 3,
  message,
  messagePlacement = "bottom",
  className,
  ...props
}: HTMLMotionProps<"div"> & BouncingDotsProps) {
  return (
    <div className={cn(bouncingDotsVariant({ messagePlacement }))}>
      <div className={cn("flex gap-2 items-center justify-center")}>
        {Array(dots)
          .fill(undefined)
          .map((_, index) => (
            <motion.div
              key={index}
              className={cn("w-3 h-3 bg-foreground rounded-full", className)}
              animate={{ y: [0, -20, 0] }}
              transition={{
                duration: 0.6,
                repeat: Number.POSITIVE_INFINITY,
                delay: index * 0.2,
                ease: "easeInOut",
              }}
              {...props}
            />
          ))}
      </div>
      {message && <div>{message}</div>}
    </div>
  )
}


demo.tsx
import { BouncingDots } from "@/components/ui/bouncing-dots";

export default function DemoOne() {
  return <BouncingDots />
}

```

Install NPM dependencies:
```bash
motion, class-variance-authority
```

Implementation Guidelines
 1. Analyze the component structure and identify all required dependencies
 2. Review the component's argumens and state
 3. Identify any required context providers or hooks and install them
 4. Questions to Ask
 - What data/props will be passed to this component?
 - Are there any specific state management requirements?
 - Are there any required assets (images, icons, etc.)?
 - What is the expected responsive behavior?
 - What is the best place to use this component in the app?

Steps to integrate
 0. Copy paste all the code above in the correct directories
 1. Install external dependencies
 2. Fill image assets with Unsplash stock images you know exist
 3. Use lucide-react icons for svgs or logos if component requires them
