import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Settings as SettingsIcon, Bell, Lock, Shield } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function Settings() {
  const { toast } = useToast();
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    sms: false,
  });
  const [security, setSecurity] = useState({
    twoFactor: false,
    sessionTimeout: true,
  });

  const handleSave = () => {
    toast({
      title: "설정 저장",
      description: "설정이 성공적으로 저장되었습니다.",
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <SettingsIcon className="text-blue-600 text-2xl mr-3" />
              <h1 className="text-xl font-bold text-gray-900">설정</h1>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-6">
          {/* 알림 설정 */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Bell className="w-5 h-5 text-blue-600" />
                <CardTitle>알림 설정</CardTitle>
              </div>
              <CardDescription>
                알림 수신 방식을 선택할 수 있습니다
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="email-notifications">이메일 알림</Label>
                  <p className="text-sm text-gray-500">
                    중요한 알림을 이메일로 받습니다
                  </p>
                </div>
                <Switch
                  id="email-notifications"
                  checked={notifications.email}
                  onCheckedChange={(checked) =>
                    setNotifications({ ...notifications, email: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="push-notifications">푸시 알림</Label>
                  <p className="text-sm text-gray-500">
                    브라우저 푸시 알림을 받습니다
                  </p>
                </div>
                <Switch
                  id="push-notifications"
                  checked={notifications.push}
                  onCheckedChange={(checked) =>
                    setNotifications({ ...notifications, push: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="sms-notifications">SMS 알림</Label>
                  <p className="text-sm text-gray-500">
                    긴급 알림을 SMS로 받습니다
                  </p>
                </div>
                <Switch
                  id="sms-notifications"
                  checked={notifications.sms}
                  onCheckedChange={(checked) =>
                    setNotifications({ ...notifications, sms: checked })
                  }
                />
              </div>
            </CardContent>
          </Card>

          {/* 보안 설정 */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Lock className="w-5 h-5 text-blue-600" />
                <CardTitle>보안 설정</CardTitle>
              </div>
              <CardDescription>
                계정 보안을 관리할 수 있습니다
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="two-factor">2단계 인증</Label>
                  <p className="text-sm text-gray-500">
                    추가 보안을 위해 2단계 인증을 활성화합니다
                  </p>
                </div>
                <Switch
                  id="two-factor"
                  checked={security.twoFactor}
                  onCheckedChange={(checked) =>
                    setSecurity({ ...security, twoFactor: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="session-timeout">세션 타임아웃</Label>
                  <p className="text-sm text-gray-500">
                    일정 시간 후 자동으로 로그아웃됩니다
                  </p>
                </div>
                <Switch
                  id="session-timeout"
                  checked={security.sessionTimeout}
                  onCheckedChange={(checked) =>
                    setSecurity({ ...security, sessionTimeout: checked })
                  }
                />
              </div>
              <div className="pt-4 border-t">
                <Button variant="outline">
                  <Shield className="w-4 h-4 mr-2" />
                  비밀번호 변경
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* 저장 버튼 */}
          <div className="flex justify-end">
            <Button onClick={handleSave} size="lg">
              설정 저장
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}









