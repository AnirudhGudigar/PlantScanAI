"use client";

import type { ReactNode } from "react";
import { BackgroundCubesGrid } from "@/components/ui/background-cubes-grid";

export function BackgroundCubesGridDemo({ children }: { children?: ReactNode }) {
  return (
    <div className="relative isolate flex min-h-[50rem] w-full items-center justify-center overflow-hidden px-5 py-24 sm:px-8 bg-black">
      <BackgroundCubesGrid className="z-0" />
      <div className="relative z-10 w-full max-w-5xl mx-auto text-center pointer-events-auto">
        {children}
      </div>
    </div>
  );
}

export default BackgroundCubesGridDemo;
