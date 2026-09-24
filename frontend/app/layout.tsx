import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial AI Agent | Financial Intelligence",
  description: "Deterministic market analytics with an AI copilot.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
