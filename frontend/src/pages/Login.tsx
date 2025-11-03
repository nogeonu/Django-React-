import { useEffect, useState } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

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
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-md bg-white shadow rounded-lg p-6">
        <h1 className="text-2xl font-semibold mb-6">로그인</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="username">아이디</Label>
            <Input id="username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="아이디" />
          </div>
          <div>
            <Label htmlFor="password">비밀번호</Label>
            <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="비밀번호" />
          </div>
          {error && <div className="text-red-600 text-sm">{error}</div>}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? '로그인 중...' : '로그인'}
          </Button>
        </form>
        <div className="mt-4 text-sm text-gray-600">
          아직 계정이 없으신가요?{' '}
          <Link className="text-blue-600 hover:underline" to="/signup">회원가입</Link>
        </div>
      </div>
    </div>
  );
}
