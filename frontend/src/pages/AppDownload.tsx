import { useEffect, useState } from "react";
import { Download, Smartphone, Shield, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ReleaseAsset {
  name: string;
  browser_download_url: string;
  size: number;
}

interface GitHubRelease {
  tag_name: string;
  name: string;
  published_at: string;
  body: string;
  assets: ReleaseAsset[];
}

export default function AppDownload() {
  const [release, setRelease] = useState<GitHubRelease | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLatestRelease = async () => {
      try {
        // Django 백엔드 프록시를 통해 GitHub API 호출
        const response = await fetch("/api/github/latest-release");
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || "최신 릴리즈를 가져올 수 없습니다.");
        }
        const data = await response.json();
        setRelease(data);
      } catch (err) {
        console.error("GitHub 릴리즈 조회 오류:", err);
        setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
      } finally {
        setLoading(false);
      }
    };

    fetchLatestRelease();
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("ko-KR", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  const apkAsset = release?.assets.find((asset) => asset.name.endsWith(".apk"));
  const ipaAsset = release?.assets.find((asset) => asset.name.endsWith(".ipa"));

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-12 max-w-5xl">
        {/* 헤더 */}
        <div className="text-center mb-12">
          <Badge className="mb-4 bg-blue-600 hover:bg-blue-700">
            모바일 앱 다운로드
          </Badge>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            건양대학교병원 환자 앱
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            언제 어디서나 간편하게 진료 예약, 건강 기록 확인, 의료진과의 소통이 가능한
            스마트 헬스케어 애플리케이션입니다.
          </p>
        </div>

        {/* 기능 소개 */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <Card className="border-2 hover:border-blue-400 transition-colors">
            <CardHeader>
              <Smartphone className="w-10 h-10 text-blue-600 mb-2" />
              <CardTitle className="text-lg">간편한 예약</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                병원 방문 없이 모바일에서 진료 예약부터 확인까지 한 번에
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-purple-400 transition-colors">
            <CardHeader>
              <Shield className="w-10 h-10 text-purple-600 mb-2" />
              <CardTitle className="text-lg">안전한 개인정보</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                암호화된 통신으로 개인 건강 정보를 안전하게 보호합니다
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-green-400 transition-colors">
            <CardHeader>
              <Zap className="w-10 h-10 text-green-600 mb-2" />
              <CardTitle className="text-lg">실시간 알림</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                예약 확인, 진료 일정 등 중요한 정보를 푸시 알림으로 전달
              </p>
            </CardContent>
          </Card>
        </div>

        {/* 다운로드 섹션 */}
        <Card className="shadow-xl border-2">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-lg">
            <CardTitle className="text-2xl flex items-center gap-2">
              <Download className="w-6 h-6" />
              최신 버전 다운로드
            </CardTitle>
            <CardDescription className="text-blue-100">
              Android APK 또는 iOS IPA 파일을 다운로드하여 설치하세요
            </CardDescription>
          </CardHeader>
          <CardContent className="p-8">
            {loading && (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <p className="mt-4 text-gray-600">최신 버전 정보를 불러오는 중...</p>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                <p className="text-red-600 font-medium">{error}</p>
                <p className="text-sm text-red-500 mt-2">
                  잠시 후 다시 시도해주세요.
                </p>
              </div>
            )}

            {!loading && !error && release && (apkAsset || ipaAsset) && (
              <div className="space-y-6">
                <div className="bg-gray-50 rounded-lg p-6 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">버전</span>
                    <Badge variant="outline" className="text-base">
                      {release.tag_name}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">배포일</span>
                    <span className="text-sm text-gray-900">
                      {formatDate(release.published_at)}
                    </span>
                  </div>
                </div>

                {release.body && (
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">
                      릴리즈 노트
                    </h3>
                    <div className="bg-blue-50 rounded-lg p-4 text-sm text-gray-700 whitespace-pre-wrap">
                      {release.body}
                    </div>
                  </div>
                )}

                {/* Android APK 다운로드 */}
                {apkAsset && (
                  <div className="space-y-3">
                    <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                      🤖 Android
                    </h3>
                    <Button
                      size="lg"
                      className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-semibold py-6 text-lg"
                      onClick={() => window.open(apkAsset.browser_download_url, "_blank")}
                    >
                      <Download className="w-5 h-5 mr-2" />
                      Android APK 다운로드 ({formatFileSize(apkAsset.size)})
                    </Button>
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-sm text-green-800">
                      <p className="font-medium mb-1">📱 Android 설치 안내</p>
                      <p>
                        APK 파일을 설치하려면 "출처를 알 수 없는 앱 설치" 권한이 필요합니다.<br />
                        설정 → 보안 → 알 수 없는 출처 허용에서 활성화해주세요.
                      </p>
                    </div>
                  </div>
                )}

                {/* iOS IPA 다운로드 */}
                {ipaAsset && (
                  <div className="space-y-3">
                    <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                      🍎 iOS
                    </h3>
                    <Button
                      size="lg"
                      variant="outline"
                      className="w-full border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 font-semibold py-6 text-lg"
                      onClick={() => window.open(ipaAsset.browser_download_url, "_blank")}
                    >
                      <Download className="w-5 h-5 mr-2" />
                      iOS IPA 다운로드 ({formatFileSize(ipaAsset.size)})
                    </Button>
                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-800">
                      <p className="font-medium mb-1">⚠️ iOS 설치 안내</p>
                      <p className="space-y-1">
                        <span className="block">• TestFlight 또는 개발자 프로비저닝 프로파일 필요</span>
                        <span className="block">• 또는 AltStore, Sideloadly 등 사이드로딩 도구 사용</span>
                        <span className="block">• 일반 사용자는 App Store 출시를 권장합니다</span>
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {!loading && !error && (!release || (!apkAsset && !ipaAsset)) && (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center space-y-4">
                <p className="text-gray-600 font-medium">
                  현재 다운로드 가능한 앱 파일이 없습니다.
                </p>
                <p className="text-sm text-gray-500">
                  새로운 버전이 곧 출시될 예정입니다.
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-left">
                  <p className="font-semibold text-blue-900 mb-2">📱 앱 준비 중</p>
                  <p className="text-blue-700">
                    Flutter 앱이 GitHub Actions를 통해 빌드되는 중이거나,<br />
                    첫 번째 릴리즈가 아직 생성되지 않았습니다.
                  </p>
                  <p className="text-blue-700 mt-2">
                    릴리즈가 생성되면 이 페이지에서 자동으로 최신 APK를 다운로드할 수 있습니다.
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => window.open("https://github.com/nogeonu/flutter-mobile/releases", "_blank")}
                  className="w-full"
                >
                  GitHub 릴리즈 페이지 확인하기 →
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 추가 정보 */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>앱 사용 중 문제가 발생하면 병원 고객센터(1234-5678)로 문의해주세요.</p>
          <p className="mt-2">
            GitHub Repository:{" "}
            <a
              href="https://github.com/nogeonu/flutter-mobile"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              nogeonu/flutter-mobile
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

