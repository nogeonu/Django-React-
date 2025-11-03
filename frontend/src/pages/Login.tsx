import { useEffect, useState } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Activity, User, Lock } from 'lucide-react';

export default function Login() {
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation() as any;
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      const target = user.role === 'medical_staff' ? '/medical_staff' : '/admin_staff';
      navigate(target, { replace: true });
    }
  }, [user, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!username || !password) {
      setError('아이디와 비밀번호를 입력해주세요.');
      return;
    }
    try {
      setLoading(true);
      const u = await login({ username, password });
      const byRole = u.role === 'medical_staff' ? '/medical_staff' : '/admin_staff';
      const redirectTo = location.state?.from?.pathname || byRole;
      navigate(redirectTo, { replace: true });
    } catch (err: any) {
      const msg = err?.response?.data?.detail || '로그인에 실패했습니다.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-black">
      {/* Left Panel - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <div className="mb-8">
            <Activity className="text-blue-500 text-3xl mb-4" />
            <h1 className="text-4xl font-bold text-white mb-2">Sign In</h1>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <Label htmlFor="username" className="text-white text-sm mb-2 block">User Name</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <Input 
                  id="username" 
                  value={username} 
                  onChange={(e) => setUsername(e.target.value)} 
                  placeholder="Enter your username"
                  className="pl-10 bg-gray-900 border-gray-700 text-white placeholder-gray-500 focus:border-blue-500"
                />
              </div>
            </div>
            
            <div>
              <Label htmlFor="password" className="text-white text-sm mb-2 block">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <Input 
                  id="password" 
                  type="password" 
                  value={password} 
                  onChange={(e) => setPassword(e.target.value)} 
                  placeholder="Enter Password"
                  className="pl-10 bg-gray-900 border-gray-700 text-white placeholder-gray-500 focus:border-blue-500"
                />
              </div>
            </div>
            
            <div className="text-right">
              <Link to="#" className="text-xs text-gray-400 hover:text-blue-500 uppercase tracking-wide">
                FORGOT PASSWORD?
              </Link>
            </div>
            
            {error && (
              <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 px-4 py-2 rounded">
                {error}
              </div>
            )}
            
            <Button 
              type="submit" 
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg uppercase tracking-wide disabled:opacity-50"
              disabled={loading}
            >
              {loading ? 'Signing In...' : 'SIGN IN'}
            </Button>
          </form>
          
          <div className="mt-6 text-center text-white text-sm">
            Don't have an account?{' '}
            <Link className="text-blue-500 hover:underline" to="/signup">
              Sign up
            </Link>
          </div>
        </div>
      </div>
      
      {/* Right Panel - Visual */}
      <div className="hidden lg:flex flex-1 bg-gradient-to-br from-gray-900 to-black relative overflow-hidden">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute inset-0" style={{
            backgroundImage: `linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%), linear-gradient(-45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%)`,
            backgroundSize: '60px 60px'
          }}></div>
        </div>
        <div className="relative z-10 p-12 flex flex-col justify-between">
          <div></div>
          <div>
            <h2 className="text-3xl font-light text-white mb-6">
              환자 정보를 한눈에,<br />
              의료 데이터를 스마트하게.
            </h2>
            <Link to="#" className="text-xs text-gray-400 hover:text-blue-500 uppercase tracking-wide">
              LEARN MORE →
            </Link>
          </div>
          <div className="flex justify-end">
            <div className="bg-white/10 backdrop-blur-sm p-4 rounded-lg flex gap-4 cursor-pointer hover:bg-white/20 transition">
              <Activity className="w-5 h-5 text-blue-400" />
              <div className="text-white text-sm">
                <div className="font-medium">의료 이미지 분석</div>
                <div className="text-xs text-gray-400">AI 기반 진단 지원</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
