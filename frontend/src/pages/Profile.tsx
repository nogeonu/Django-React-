import { useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { User, Mail, Building2, Shield, Save } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/api";

export default function Profile() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    first_name: user?.first_name || '',
    last_name: user?.last_name || '',
    email: user?.email || '',
    department: user?.department || '',
  });

  const handleSave = async () => {
    try {
      // TODO: API 엔드포인트 구현 필요
      await apiRequest('PATCH', '/api/auth/user/', formData);
      toast({
        title: "프로필 업데이트",
        description: "프로필이 성공적으로 업데이트되었습니다.",
      });
      setIsEditing(false);
    } catch (error: any) {
      toast({
        title: "오류 발생",
        description: error?.response?.data?.error || "프로필 업데이트 중 오류가 발생했습니다.",
        variant: "destructive",
      });
    }
  };

  const getRoleLabel = (role: string) => {
    switch (role) {
      case 'medical_staff':
        return '의료진';
      case 'admin_staff':
        return '원무과';
      case 'superuser':
        return '관리자';
      default:
        return role;
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card>
          <CardContent className="pt-6">
            <p className="text-gray-500">로그인이 필요합니다.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <User className="text-blue-600 text-2xl mr-3" />
              <h1 className="text-xl font-bold text-gray-900">마이페이지</h1>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <CardHeader>
            <CardTitle>프로필 정보</CardTitle>
            <CardDescription>
              개인 정보를 확인하고 수정할 수 있습니다
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 사용자 정보 */}
            <div className="flex items-center space-x-4 pb-6 border-b">
              <div className="w-20 h-20 rounded-full bg-blue-100 flex items-center justify-center">
                <User className="w-10 h-10 text-blue-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  {user.first_name || user.last_name 
                    ? `${user.last_name || ''}${user.first_name || ''}`.trim()
                    : user.username}
                </h2>
                <p className="text-gray-500">{getRoleLabel(user.role)}</p>
                {user.department && (
                  <p className="text-sm text-gray-500">{user.department}</p>
                )}
              </div>
            </div>

            {/* 편집 가능한 정보 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="username">사용자명</Label>
                <Input
                  id="username"
                  value={user.username}
                  disabled
                  className="mt-1 bg-gray-50"
                />
                <p className="text-xs text-gray-500 mt-1">사용자명은 변경할 수 없습니다</p>
              </div>

              <div>
                <Label htmlFor="email">이메일</Label>
                {isEditing ? (
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="mt-1"
                  />
                ) : (
                  <div className="mt-1 flex items-center space-x-2 p-2 bg-gray-50 rounded-md">
                    <Mail className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-900">{user.email}</span>
                  </div>
                )}
              </div>

              <div>
                <Label htmlFor="last_name">성</Label>
                {isEditing ? (
                  <Input
                    id="last_name"
                    value={formData.last_name}
                    onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                    className="mt-1"
                  />
                ) : (
                  <div className="mt-1 p-2 bg-gray-50 rounded-md">
                    <span className="text-gray-900">{user.last_name || '-'}</span>
                  </div>
                )}
              </div>

              <div>
                <Label htmlFor="first_name">이름</Label>
                {isEditing ? (
                  <Input
                    id="first_name"
                    value={formData.first_name}
                    onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                    className="mt-1"
                  />
                ) : (
                  <div className="mt-1 p-2 bg-gray-50 rounded-md">
                    <span className="text-gray-900">{user.first_name || '-'}</span>
                  </div>
                )}
              </div>

              <div>
                <Label htmlFor="department">진료과/부서</Label>
                {isEditing ? (
                  <Input
                    id="department"
                    value={formData.department}
                    onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                    className="mt-1"
                    placeholder="예: 호흡기내과"
                  />
                ) : (
                  <div className="mt-1 flex items-center space-x-2 p-2 bg-gray-50 rounded-md">
                    <Building2 className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-900">{user.department || '-'}</span>
                  </div>
                )}
              </div>

              <div>
                <Label htmlFor="role">역할</Label>
                <div className="mt-1 flex items-center space-x-2 p-2 bg-gray-50 rounded-md">
                  <Shield className="w-4 h-4 text-gray-400" />
                  <span className="text-gray-900">{getRoleLabel(user.role)}</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">역할은 변경할 수 없습니다</p>
              </div>
            </div>

            {/* 액션 버튼 */}
            <div className="flex justify-end space-x-3 pt-6 border-t">
              {isEditing ? (
                <>
                  <Button variant="outline" onClick={() => {
                    setIsEditing(false);
                    setFormData({
                      first_name: user?.first_name || '',
                      last_name: user?.last_name || '',
                      email: user?.email || '',
                      department: user?.department || '',
                    });
                  }}>
                    취소
                  </Button>
                  <Button onClick={handleSave}>
                    <Save className="w-4 h-4 mr-2" />
                    저장
                  </Button>
                </>
              ) : (
                <Button onClick={() => setIsEditing(true)}>
                  정보 수정
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}


