import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
  xsrfCookieName: 'csrftoken',
  xsrfHeaderName: 'X-CSRFToken',
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
      headers: data instanceof FormData ? {
        'Content-Type': 'multipart/form-data',
      } : undefined,
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

export const registerApi = async (data: { username: string; password: string; email?: string; department?: string; first_name?: string; last_name?: string }) => {
  const res = await apiClient.post('/api/auth/register', data);
  return res.data;
};

export const patientLoginApi = async (data: { account_id: string; password: string }) => {
  const res = await apiClient.post('/api/patients/login/', data);
  return res.data;
};

export const getDoctorsApi = async (department?: string) => {
  const query = department ? `?department=${encodeURIComponent(department)}` : '';
  const res = await apiClient.get(`/api/auth/doctors/${query}`);
  return res.data;
};

export const getPatientProfile = async (accountId: string) => {
  const res = await apiClient.get(`/api/patients/profile/${accountId}/`);
  return res.data;
};

export const updatePatientProfile = async (accountId: string, data: Record<string, unknown>) => {
  const res = await apiClient.put(`/api/patients/profile/${accountId}/`, data);
  return res.data;
};

export const patientSignupApi = async (data: { account_id: string; name: string; email: string; phone: string; password: string }) => {
  const res = await apiClient.post('/api/patients/signup/', data);
  return res.data;
};

export const getAppointmentsApi = async (params?: Record<string, unknown>) => {
  const res = await apiClient.get('/api/patients/appointments/', { params });
  return res.data;
};

export const getPatientsApi = async (
  params?: Record<string, unknown>,
) => {
  const res = await apiClient.get('/api/patients/patients/', {
    params,
  });
  return res.data;
};

export const searchPatientsApi = async (searchTerm: string) => {
  const res = await apiClient.get('/api/patients/patients/', {
    params: { search: searchTerm },
  });
  return res.data;
};

export const createAppointmentApi = async (data: Record<string, unknown>) => {
  const res = await apiClient.post('/api/patients/appointments/', data);
  return res.data;
};

export const updateAppointmentApi = async (id: string, data: Record<string, unknown>) => {
  const res = await apiClient.patch(`/api/patients/appointments/${id}/`, data);
  return res.data;
};

export const deleteAppointmentApi = async (id: string) => {
  const res = await apiClient.delete(`/api/patients/appointments/${id}/`);
  return res.data;
};

export default apiClient;
