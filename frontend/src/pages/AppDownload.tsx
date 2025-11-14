import { useEffect, useState } from "react";
import { Download, Smartphone, Shield, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* 헤더 */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <Download className="w-8 h-8 text-blue-600" />
            <div>
              <p className="text-sm font-semibold uppercase tracking-wider text-blue-600">
                MOBILE APP DOWNLOAD
              </p>
              <h1 className="text-3xl font-bold text-gray-900">
                건양대학교병원 환자 앱
              </h1>
            </div>
          </div>
          <p className="text-base text-gray-600 leading-relaxed">
            언제 어디서나 간편하게 진료 예약, 건강 기록 확인, 의료진과의 소통이 가능한 스마트 헬스케어 애플리케이션입니다.
          </p>
        </div>

        {/* 기능 소개 */}
        <div className="grid md:grid-cols-3 gap-4 mb-8">
          <Card className="border border-gray-200 bg-white hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <Smartphone className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-bold text-gray-900 mb-1">간편한 예약</h3>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    병원 방문 없이 모바일에서 진료 예약부터 확인까지 한 번에
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border border-gray-200 bg-white hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-purple-50 rounded-lg">
                  <Shield className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <h3 className="font-bold text-gray-900 mb-1">안전한 개인정보</h3>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    암호화된 통신으로 개인 건강 정보를 안전하게 보호합니다
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border border-gray-200 bg-white hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-green-50 rounded-lg">
                  <Zap className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <h3 className="font-bold text-gray-900 mb-1">실시간 알림</h3>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    예약 확인, 진료 일정 등 중요한 정보를 푸시 알림으로 전달
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 다운로드 섹션 */}
        <Card className="border border-gray-200 bg-white shadow-sm">
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
                {/* 버전 정보 헤더 */}
                <div className="flex items-start gap-3 pb-6 border-b border-gray-200">
                  <Download className="w-6 h-6 text-blue-600 mt-1" />
                  <div className="flex-1">
                    <h2 className="text-xl font-bold text-gray-900 mb-2">
                      최신 버전 다운로드
                    </h2>
                    <div className="flex items-center gap-4 text-sm">
                      <div className="flex items-center gap-2">
                        <span className="text-gray-600">버전</span>
                        <Badge variant="outline" className="font-semibold">
                          {release.tag_name}
                        </Badge>
                      </div>
                      <div className="text-gray-600">
                        배포일: <span className="text-gray-900 font-medium">{formatDate(release.published_at)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {release.body && (
                  <div className="bg-blue-50 border-l-4 border-blue-600 p-6">
                    <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                      <span className="text-blue-600">📋</span>
                      릴리즈 노트
                    </h3>
                    <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                      {release.body}
                    </div>
                  </div>
                )}

                {/* Android APK 다운로드 */}
                {apkAsset && (
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-3 bg-green-50 rounded-lg">
                        <span className="text-2xl">🤖</span>
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-900 text-lg">Android</h3>
                        <p className="text-sm text-gray-600">APK 파일 · {formatFileSize(apkAsset.size)}</p>
                      </div>
                    </div>
                    <Button
                      size="lg"
                      className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-4 text-base rounded-lg"
                      onClick={() => window.open(apkAsset.browser_download_url, "_blank")}
                    >
                      <Download className="w-5 h-5 mr-2" />
                      Android APK 다운로드
                    </Button>
                    <div className="mt-4 bg-green-50 border-l-4 border-green-600 p-4">
                      <p className="text-sm font-semibold text-green-900 mb-2">📱 Android 설치 안내</p>
                      <p className="text-sm text-green-800 leading-relaxed">
                        APK 파일을 설치하려면 "출처를 알 수 없는 앱 설치" 권한이 필요합니다.
                        설정 → 보안 → 알 수 없는 출처 허용에서 활성화해주세요.
                      </p>
                    </div>
                  </div>
                )}

                {/* iOS IPA 다운로드 */}
                {ipaAsset && (
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <span className="text-2xl">🍎</span>
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-900 text-lg">iOS</h3>
                        <p className="text-sm text-gray-600">IPA 파일 · {formatFileSize(ipaAsset.size)}</p>
                      </div>
                    </div>
                    <Button
                      size="lg"
                      variant="outline"
                      className="w-full border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 font-semibold py-4 text-base rounded-lg"
                      onClick={() => window.open(ipaAsset.browser_download_url, "_blank")}
                    >
                      <Download className="w-5 h-5 mr-2" />
                      iOS IPA 다운로드
                    </Button>
                    <div className="mt-4 bg-amber-50 border-l-4 border-amber-600 p-4">
                      <p className="text-sm font-semibold text-amber-900 mb-2">⚠️ iOS 설치 안내</p>
                      <ul className="text-sm text-amber-800 leading-relaxed space-y-1">
                        <li>• TestFlight 또는 개발자 프로비저닝 프로파일 필요</li>
                        <li>• 또는 AltStore, Sideloadly 등 사이드로딩 도구 사용</li>
                        <li>• 일반 사용자는 App Store 출시를 권장합니다</li>
                      </ul>
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

        {/* 하단 정보 */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-blue-50 rounded-lg">
              <span className="text-xl">ℹ️</span>
            </div>
            <div className="flex-1 text-sm text-gray-600 leading-relaxed">
              <p className="font-semibold text-gray-900 mb-2">
                건양대학교 병원 · 환자 포털 서비스
              </p>
              <p>
                앱 사용 중 문제가 발생하면 병원 고객센터(1234-5678)로 문의해주세요.
              </p>
              <p className="mt-2">
                GitHub Repository:{" "}
                <a
                  href="https://github.com/nogeonu/flutter-mobile"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline font-medium"
                >
                  nogeonu/flutter-mobile
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

