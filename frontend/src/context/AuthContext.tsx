import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { loginApi, meApi, logoutApi } from '@/lib/api';

export type User = {
  id: number;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
  role: 'medical_staff' | 'admin_staff' | 'superuser';
};

type AuthContextType = {
  user: User | null;
  loading: boolean;
  login: (params: { username: string; password: string }) => Promise<User>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      try {
        const me = await meApi();
        setUser(me);
      } catch {
        setUser(null);
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  const value = useMemo<AuthContextType>(() => ({
    user,
    loading,
    login: async ({ username, password }) => {
      const u = await loginApi({ username, password });
      setUser(u);
      return u;
    },
    logout: async () => {
      await logoutApi();
      setUser(null);
    },
  }), [user, loading]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
