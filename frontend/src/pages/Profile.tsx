import { useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
    <div className="space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">마이페이지</h1>
        <p className="text-muted-foreground">개인 정보를 확인하고 수정할 수 있습니다</p>
      </div>
        <Card className="border-none shadow-md">
          <CardHeader className="border-b border-slate-200 dark:border-slate-800">
            <CardTitle>프로필 정보</CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            {/* 사용자 정보 */}
            <div className="flex items-center space-x-4 pb-6 border-b border-slate-200 dark:border-slate-800 mb-6">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary to-accent text-white flex items-center justify-center">
                <User className="w-8 h-8" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-foreground">
                  {user.first_name || user.last_name 
                    ? `${user.last_name || ''}${user.first_name || ''}`.trim()
                    : user.username}
                </h2>
                <p className="text-muted-foreground">{getRoleLabel(user.role)}</p>
                {user.department && (
                  <p className="text-sm text-muted-foreground">{user.department}</p>
                )}
              </div>
            </div>

            {/* 편집 가능한 정보 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label htmlFor="username" className="text-sm font-medium text-foreground">사용자명</Label>
                <Input
                  id="username"
                  value={user.username}
                  disabled
                  className="bg-slate-50 dark:bg-slate-800"
                />
                <p className="text-xs text-muted-foreground">사용자명은 변경할 수 없습니다</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-medium text-foreground">이메일</Label>
                {isEditing ? (
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  />
                ) : (
                  <div className="flex items-center space-x-2 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <Mail className="w-4 h-4 text-muted-foreground" />
                    <span className="text-foreground">{user.email}</span>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="last_name" className="text-sm font-medium text-foreground">성</Label>
                {isEditing ? (
                  <Input
                    id="last_name"
                    value={formData.last_name}
                    onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                  />
                ) : (
                  <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <span className="text-foreground">{user.last_name || '-'}</span>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="first_name" className="text-sm font-medium text-foreground">이름</Label>
                {isEditing ? (
                  <Input
                    id="first_name"
                    value={formData.first_name}
                    onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                  />
                ) : (
                  <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <span className="text-foreground">{user.first_name || '-'}</span>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="department" className="text-sm font-medium text-foreground">진료과/부서</Label>
                {isEditing ? (
                  <Input
                    id="department"
                    value={formData.department}
                    onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                    placeholder="예: 호흡기내과"
                  />
                ) : (
                  <div className="flex items-center space-x-2 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <Building2 className="w-4 h-4 text-muted-foreground" />
                    <span className="text-foreground">{user.department || '-'}</span>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="role" className="text-sm font-medium text-foreground">역할</Label>
                <div className="flex items-center space-x-2 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <Shield className="w-4 h-4 text-muted-foreground" />
                  <span className="text-foreground">{getRoleLabel(user.role)}</span>
                </div>
                <p className="text-xs text-muted-foreground">역할은 변경할 수 없습니다</p>
              </div>
            </div>

            {/* 액션 버튼 */}
            <div className="flex justify-end space-x-3 pt-6 border-t border-slate-200 dark:border-slate-800">
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
                  <Button onClick={handleSave} className="bg-primary hover:bg-primary/90">
                    <Save className="w-4 h-4 mr-2" />
                    저장
                  </Button>
                </>
              ) : (
                <Button onClick={() => setIsEditing(true)} className="bg-primary hover:bg-primary/90">
                  정보 수정
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
    </div>
  );
}





