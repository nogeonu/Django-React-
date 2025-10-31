import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { registerApi } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

export default function Signup() {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'medical_staff' | 'admin_staff' | ''>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!username || !password || !role) {
      setError('아이디/비밀번호/역할을 모두 입력해주세요.');
      return;
    }
    try {
      setLoading(true);
      await registerApi({
        username,
        password,
        email,
        role,
        first_name: firstName || undefined,
        last_name: lastName || undefined,
      });
      navigate('/login', { replace: true });
    } catch (err: any) {
      const msg = err?.response?.data?.detail || '회원가입에 실패했습니다.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-md bg-white shadow rounded-lg p-6">
        <h1 className="text-2xl font-semibold mb-6">회원가입</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="username">아이디</Label>
            <Input id="username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="아이디" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <Label htmlFor="lastName">성</Label>
              <Input id="lastName" value={lastName} onChange={(e) => setLastName(e.target.value)} placeholder="성" />
            </div>
            <div>
              <Label htmlFor="firstName">이름</Label>
              <Input id="firstName" value={firstName} onChange={(e) => setFirstName(e.target.value)} placeholder="이름" />
            </div>
          </div>
          <div>
            <Label htmlFor="email">이메일</Label>
            <Input id="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="이메일(선택)" />
          </div>
          <div>
            <Label htmlFor="password">비밀번호</Label>
            <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="비밀번호" />
          </div>
          <div>
            <Label>역할</Label>
            <Select value={role} onValueChange={(v) => setRole(v as any)}>
              <SelectTrigger>
                <SelectValue placeholder="역할을 선택하세요" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="medical_staff">의료진</SelectItem>
                <SelectItem value="admin_staff">원무과</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {error && <div className="text-red-600 text-sm">{error}</div>}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? '가입 중...' : '가입하기'}
          </Button>
        </form>
      </div>
    </div>
  );
}
