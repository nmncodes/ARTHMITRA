'use client'

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { ArrowLeft, BrainCircuit, TrendingUp, ShieldCheck, AlertCircle, Target, Wallet, PieChart as PieChartIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend } from 'recharts';
import RiskComparison from '@/components/RiskComparison';

// Colors for the Pie Chart
const COLORS = ['#2563eb', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444'];
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function GoalPlanner() {
  const router = useRouter();
  const [target, setTarget] = useState<number>(100000);
  const [years, setYears] = useState<number>(3);
  const [sector, setSector] = useState<string>("Lifestyle");
  const [risk, setRisk] = useState<string>("Moderate");
  const [loading, setLoading] = useState(false);
  const [aiResult, setAiResult] = useState<any>(null);

  const runAudit = async () => {
    setLoading(true);
    try {
      // Temporarily simulating the NEW backend response structure for testing the UI
      // In production, you will fetch this from your Python backend
            const response = await fetch(`${API_BASE_URL}/api/finance/check-goal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal_type: sector, amount: target, years: years, risk_profile: risk }),
      });
      
      const data = await response.json();
      
      // If your backend isn't returning the new structure yet, we'll map the old one to the new one safely
      const formattedResult = {
        future_value: data.future_value || target * 1.2,
        sip_amount: data.sip_amount || 3500,
        advice: data.advice || "Based on your timeline, a balanced approach is best.",
        // We expect the backend to send an array for the pie chart: [{name: "Equity", value: 60, amount: 2100}]
        allocation: data.allocation_chart || [
            { name: "Debt Funds", value: 70, amount: (data.sip_amount || 3500) * 0.7 },
            { name: "Fixed Deposits", value: 30, amount: (data.sip_amount || 3500) * 0.3 }
        ],
        // We expect specific schemes from the backend
        specific_schemes: data.specific_schemes || [
            { name: "High-Grade Corporate Bond Fund", category: "Debt", amount: (data.sip_amount || 3500) * 0.7, reason: "Stable returns for short-term goals." },
            { name: "Top Bank 3-Year FD / RD", category: "Safe", amount: (data.sip_amount || 3500) * 0.3, reason: "Zero market risk, guaranteed capital protection." }
        ]
      };
      
      setAiResult(formattedResult);
    } catch (err) {
      console.error("Audit failed", err);
    } finally {
      setLoading(false);
    }
  };

  // Custom Tooltip for the Pie Chart
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-bold text-sm text-foreground">{payload[0].name}</p>
          <p className="text-primary font-semibold text-sm">
            {payload[0].value}% (₹{payload[0].payload.amount.toLocaleString('en-IN')}/mo)
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="container mx-auto p-4 md:p-6 max-w-6xl space-y-8">
      <div className="flex items-center gap-3 border-b pb-4">
        <Button variant="ghost" size="icon" onClick={() => router.back()} className="shrink-0">
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
            <Target className="text-primary w-6 h-6" />
        </div>
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">Smart Goal Planner</h1>
          <p className="text-sm text-muted-foreground">AI-driven investment strategies tailored to Indian markets</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-12">
        {/* LEFT COLUMN: Input Form */}
        <div className="lg:col-span-4 space-y-6">
            <Card className="border-border/40 shadow-sm sticky top-24">
            <CardHeader className="bg-slate-50 border-b border-border/40 rounded-t-xl pb-4">
                <CardTitle className="text-lg flex items-center gap-2">
                    <BrainCircuit className="w-5 h-5 text-primary" /> 
                    Goal Parameters
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5 pt-6">
                <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground">Goal Category</label>
                <Select onValueChange={setSector} defaultValue={sector}>
                    <SelectTrigger className="bg-white"><SelectValue /></SelectTrigger>
                    <SelectContent>
                    <SelectItem value="Education">📚 Higher Education</SelectItem>
                    <SelectItem value="Medical">🏥 Specialized Medical</SelectItem>
                    <SelectItem value="Lifestyle">🚗 Car / Travel</SelectItem>
                    <SelectItem value="Home">🏠 House Downpayment</SelectItem>
                    <SelectItem value="Retirement">👴 Retirement Corpus</SelectItem>
                    </SelectContent>
                </Select>
                </div>
                
                <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground">Today's Cost (₹)</label>
                <Input type="number" value={target} onChange={(e) => setTarget(Number(e.target.value))} className="font-medium bg-white" />
                </div>
                
                <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground">Time Horizon (Years)</label>
                <Input type="number" value={years} onChange={(e) => setYears(Number(e.target.value))} className="bg-white" />
                </div>

                <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground">Your Risk Appetite</label>
                <Select onValueChange={setRisk} defaultValue={risk}>
                    <SelectTrigger className="bg-white"><SelectValue /></SelectTrigger>
                    <SelectContent>
                    <SelectItem value="Conservative">Conservative (Safe, Low Return)</SelectItem>
                    <SelectItem value="Moderate">Moderate (Balanced Risk)</SelectItem>
                    <SelectItem value="Aggressive">Aggressive (High Risk & Return)</SelectItem>
                    </SelectContent>
                </Select>
                </div>

                <Button onClick={runAudit} className="w-full h-12 text-base font-bold shadow-md hover:-translate-y-0.5 transition-all" disabled={loading}>
                {loading ? "Generating Strategy..." : "Generate Investment Plan"}
                </Button>
            </CardContent>
            </Card>
        </div>

        {/* RIGHT COLUMN: AI Results OR Empty State */}
        <div className="lg:col-span-8">
            {aiResult ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                {/* Top Metrics Row */}
                <div className="grid md:grid-cols-2 gap-4">
                    <Card className="bg-gradient-to-br from-blue-50 to-white border-blue-100 shadow-sm">
                        <CardContent className="p-6">
                            <div className="flex items-center gap-2 text-blue-800 mb-2">
                                <TrendingUp size={18}/> 
                                <span className="text-sm font-bold uppercase tracking-wider">Inflation Adjusted Target</span>
                            </div>
                            <p className="text-xs text-muted-foreground mb-1">Estimated cost in {years} years</p>
                            <h2 className="text-3xl font-black text-blue-950">₹{aiResult.future_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</h2>
                        </CardContent>
                    </Card>

                    <Card className="bg-gradient-to-br from-emerald-50 to-white border-emerald-100 shadow-sm relative overflow-hidden">
                        <div className="absolute right-0 top-0 opacity-10 scale-150 translate-x-4 -translate-y-4">
                            <Wallet size={100} />
                        </div>
                        <CardContent className="p-6 relative z-10">
                            <div className="flex items-center gap-2 text-emerald-800 mb-2">
                                <Target size={18}/> 
                                <span className="text-sm font-bold uppercase tracking-wider">Required Monthly SIP</span>
                            </div>
                            <p className="text-xs text-muted-foreground mb-1">Invest this much every month</p>
                            <h2 className="text-3xl font-black text-emerald-950">₹{aiResult.sip_amount.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</h2>
                        </CardContent>
                    </Card>
                </div>

                {/* AI Advice & Chart Row */}
                <Card className="border-border/40 shadow-sm">
                    <CardHeader className="border-b border-border/40 pb-4">
                        <CardTitle className="flex items-center gap-2 text-lg">
                            <ShieldCheck className="text-primary w-5 h-5" /> 
                            AI Strategy & Asset Allocation
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                        <div className="grid md:grid-cols-2 divide-y md:divide-y-0 md:divide-x border-border/40">
                            {/* AI Text Advice */}
                            <div className="p-6">
                                <p className="text-sm leading-relaxed text-muted-foreground">
                                    {aiResult.advice}
                                </p>
                            </div>
                            
                            {/* Pie Chart */}
                            <div className="p-6 flex flex-col items-center justify-center min-h-[250px]">
                                <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wide mb-4 flex items-center gap-2">
                                    <PieChartIcon size={14} /> Portfolio Split
                                </h4>
                                <div className="w-full h-[200px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={aiResult.allocation}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="value"
                                            >
                                                {aiResult.allocation.map((entry: any, index: number) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                            <RechartsTooltip content={<CustomTooltip />} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="flex flex-wrap justify-center gap-3 mt-2">
                                    {aiResult.allocation.map((item: any, idx: number) => (
                                        <div key={idx} className="flex items-center gap-1.5 text-xs font-medium">
                                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></div>
                                            {item.name}: {item.value}%
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Actionable Schemes Row */}
                <div className="space-y-4">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        💡 Where to invest your ₹{aiResult.sip_amount.toLocaleString('en-IN', { maximumFractionDigits: 0 })}/month
                    </h3>
                    <div className="grid sm:grid-cols-2 gap-4">
                        {aiResult.specific_schemes.map((scheme: any, idx: number) => (
                            <Card key={idx} className="border-border/40 hover:border-primary/40 hover:shadow-md transition-all group">
                                <CardContent className="p-5">
                                    <div className="flex justify-between items-start mb-3">
                                        <div>
                                            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground bg-muted px-2 py-1 rounded-md mb-2 inline-block">
                                                {scheme.category}
                                            </span>
                                            <h4 className="font-bold text-foreground text-sm group-hover:text-primary transition-colors">{scheme.name}</h4>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-xs text-muted-foreground">Monthly Installment</p>
                                            <p className="font-black text-primary text-lg">₹{scheme.amount.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</p>
                                        </div>
                                    </div>
                                    <p className="text-xs text-muted-foreground leading-relaxed border-t border-border/40 pt-3">
                                        {scheme.reason}
                                    </p>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>

                {/* MOVED HERE: Risk Comparison shows at the bottom of the generated results */}
                <div className="pt-4">
                    <RiskComparison 
                        specificSchemes={aiResult.specific_schemes}
                        userRiskProfile={risk}
                    />
                </div>

            </div>
            ) : (
            <div className="space-y-8 h-full flex items-center justify-center min-h-[500px]">
                {/* NEW CLEAN EMPTY STATE */}
                <div className="flex flex-col items-center justify-center border-2 border-dashed border-border/60 bg-slate-50/50 rounded-2xl p-10 text-muted-foreground w-full">
                    <div className="w-16 h-16 bg-primary/5 rounded-full flex items-center justify-center mb-6">
                        <BrainCircuit size={32} className="text-primary/40" />
                    </div>
                    <h3 className="text-xl font-bold text-foreground mb-2">Ready to plan your future?</h3>
                    <p className="text-center max-w-md text-sm leading-relaxed">
                        Adjust your Goal Parameters on the left and click Generate. 
                        Arth-Mitra will build a custom portfolio and provide a complete stress-test analysis of your crash risk.
                    </p>
                </div>
            </div>
            )}
        </div>
      </div>
    </div>
  );
}