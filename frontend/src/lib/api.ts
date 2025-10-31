import axios from 'axios';

const API_BASE_URL = '';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      console.error('Unauthorized access');
    }
    return Promise.reject(error);
  }
);

export const apiRequest = async (method: string, url: string, data?: any) => {
  try {
    const response = await apiClient.request({
      method,
      url,
      data,
    });
    return response.data;
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
};

export const loginApi = async (data: { username: string; password: string }) => {
  const res = await apiClient.post('/api/auth/login', data);
  return res.data;
};

export const meApi = async () => {
  const res = await apiClient.get('/api/auth/me');
  return res.data;
};

export const logoutApi = async () => {
  const res = await apiClient.post('/api/auth/logout');
  return res.data;
};

export const registerApi = async (data: { username: string; password: string; email?: string; role?: string; first_name?: string; last_name?: string }) => {
  const res = await apiClient.post('/api/auth/register', data);
  return res.data;
};

export default apiClient;
