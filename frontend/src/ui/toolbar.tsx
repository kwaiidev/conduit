import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Mic, 
  MicOff, 
  Keyboard, 
  Zap, 
  Play, 
  Settings, 
  Power,
  ChevronRight,
  Info
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function CompactToolbar() {
  const navigate = useNavigate();
  const [leftJaw, setLeftJaw] = useState(true);
  const [rightJaw, setRightJaw] = useState(true);
  const [bothJaws, setBothJaws] = useState(false);
  const [voiceActive, setVoiceActive] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [isPassive, setIsPassive] = useState(true);

  return (
    <motion.div 
      layout
      className="relative bg-white/80 dark:bg-[#121218]/90 backdrop-blur-3xl border border-slate-200 dark:border-white/10 rounded-[40px] shadow-2xl flex items-center p-2 gap-2"
    >
      {/* Signal Status */}
      <div className="flex items-center gap-3 px-6 py-4 border-r border-slate-100 dark:border-white/5">
        <Zap size={20} className="text-[#FF2D8D] fill-[#FF2D8D]/20" />
        <div className="flex flex-col">
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-white/40">Signal</span>
          <span className="text-sm font-black text-slate-900 dark:text-white">94%</span>
        </div>
      </div>

      {/* EEG Jaw Controls */}
      <div className="flex items-center gap-1.5 px-3">
        <div className="flex items-center gap-1 bg-slate-50 dark:bg-white/5 p-1 rounded-3xl">
          <ToolbarButton active={leftJaw} onClick={() => setLeftJaw(!leftJaw)} label="L-Jaw" icon="L" />
          <ToolbarButton active={bothJaws} onClick={() => setBothJaws(!bothJaws)} label="M-Jaw" icon="M" />
          <ToolbarButton active={rightJaw} onClick={() => setRightJaw(!rightJaw)} label="R-Jaw" icon="R" />
        </div>

        <div className="w-px h-8 bg-slate-100 dark:bg-white/5 mx-1" />

        <ToolbarButton 
          active={voiceActive} 
          onClick={() => setVoiceActive(!voiceActive)} 
          label="Voice" 
          icon={voiceActive ? <Mic size={18} /> : <MicOff size={18} />}
          color="blue"
        />
      </div>

      {/* Safety Mode Toggle */}
      <button 
        onClick={() => setIsPassive(!isPassive)}
        className={`px-4 py-2 rounded-2xl flex items-center gap-2 transition-all border ${
          isPassive ? 'bg-amber-500/10 border-amber-500/20 text-amber-600' : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-600'
        }`}
      >
        <div className={`w-2 h-2 rounded-full ${isPassive ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'}`} />
        <span className="text-[10px] font-bold uppercase tracking-wider">{isPassive ? 'Passive' : 'Active'}</span>
      </button>

      {/* Onboarding Hover Trigger */}
      <div onMouseEnter={() => setIsHovered(true)} onMouseLeave={() => setIsHovered(false)}>
        <motion.div 
          animate={{ width: isHovered ? 240 : 64 }}
          className="h-16 flex items-center justify-center rounded-[32px] overflow-hidden cursor-help"
        >
          <AnimatePresence mode="wait">
            {isHovered ? (
              <motion.div key="exp" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-4 px-6 w-full">
                <div className="w-10 h-10 rounded-full bg-[#FF2D8D] flex items-center justify-center text-white"><Play size={18} /></div>
                <div className="flex flex-col text-left"><span className="text-xs font-bold dark:text-white">Launch Onboarding</span></div>
                <button onClick={() => navigate('/training/placement')} className="ml-auto"><ChevronRight size={16} /></button>
              </motion.div>
            ) : (
              <Info size={22} className="text-slate-400" />
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      <div className="flex items-center gap-1 pr-3">
         <Settings size={18} className="text-slate-400 mx-2" />
         <button className="w-12 h-12 rounded-full bg-slate-900 dark:bg-white text-white dark:text-[#121218] flex items-center justify-center shadow-lg"><Power size={18} /></button>
      </div>
    </motion.div>
  );
};

const ToolbarButton: React.FC<{ active: boolean; onClick: () => void; label: string; icon: any; color?: string }> = ({ active, onClick, label, icon, color }) => (
  <button 
    onClick={onClick}
    className={`w-14 h-14 rounded-full flex flex-col items-center justify-center transition-all ${
      active ? (color === 'blue' ? 'bg-indigo-500 text-white' : 'bg-slate-900 dark:bg-white text-white dark:text-slate-900') : 'text-slate-400'
    }`}
  >
    {icon}
    <span className="text-[8px] font-bold uppercase mt-1">{label}</span>
  </button>
);