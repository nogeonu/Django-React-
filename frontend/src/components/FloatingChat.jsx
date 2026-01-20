import React, { useEffect, useMemo, useRef, useState } from 'react';
import './FloatingChat.css';

const CHAT_SERVER = ''; // 로컬 서버 연동을 위해 비움

const buildWsBaseUrl = (serverUrl) => {
    // 1. Explicit server URL (e.g., hardcoded or from env)
    if (serverUrl) {
        try {
            const parsed = new URL(serverUrl);
            const scheme = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${scheme}//${parsed.host}`;
        } catch (err) { }
    }

    // 2. Fallback to current window location
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.host; // e.g., 'localhost:5173' or '34.42.223.43'

    // 개발 환경 (Vite는 다른 포트 사용)
    if (host.includes('localhost') || host.includes('127.0.0.1')) {
        // 로컬 개발 시 Django 서버 포트로 연결
        if (host.includes(':')) {
            const [hostname, port] = host.split(':');
            // Vite 포트(5173)를 Django 포트(8000)로 변경
            if (port === '5173' || port === '3000') {
                host = `${hostname}:8000`;
            }
        } else {
            host = `${host}:8000`;
        }
    } else {
        // 프로덕션 환경: Nginx가 /ws를 프록시하므로 같은 호스트 사용
        // 포트 없이 사용 (Nginx가 80 포트에서 /ws를 8000으로 프록시)
        // host는 그대로 사용 (예: '34.42.223.43')
    }

    return `${scheme}//${host}`;
};

const API_BASE_URL = CHAT_SERVER;
const WS_BASE_URL = buildWsBaseUrl(CHAT_SERVER);

const buildApiUrl = (path) => (API_BASE_URL ? `${API_BASE_URL}${path}` : path);

const getCookie = (name) => {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return undefined;
};

const formatTime = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    let hours = date.getHours();
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const ampm = hours >= 12 ? '오후' : '오전';
    hours %= 12;
    hours = hours || 12;
    return `${ampm} ${hours}:${minutes}`;
};

const formatDateLabel = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const dateStr = `${date.getFullYear()}년 ${date.getMonth() + 1}월 ${date.getDate()}일`;
    const dayStr = days[date.getDay()];
    return `${dateStr} ${dayStr}요일`;
};

const isImage = (filename) => (/\\.(jpg|jpeg|png|gif|webp|bmp)$/i).test(filename || '');

const getRoomLabel = (room) => {
    const name = room?.name || '';
    if (name.startsWith('case:dm:')) return '1:1 대화';
    return name.replace(/^(case|dept|shift):/, '');
};

const FloatingChat = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [unreadCount, setUnreadCount] = useState(0);
    const [currentUser, setCurrentUser] = useState(null);
    const [currentTab, setCurrentTab] = useState('friends');
    const [friends, setFriends] = useState([]);
    const [rooms, setRooms] = useState([]);
    const [currentRoom, setCurrentRoom] = useState(null);
    const [messages, setMessages] = useState([]);
    const [headerTitle, setHeaderTitle] = useState('의료진 채팅');
    const [headerStatus, setHeaderStatus] = useState('대기 중');
    const [messageInput, setMessageInput] = useState('');
    const [isSelectionMode, setIsSelectionMode] = useState(false);
    const [selectedUserIds, setSelectedUserIds] = useState(new Set());

    const chatMessagesRef = useRef(null);
    const messageInputRef = useRef(null);
    const fileInputRef = useRef(null);
    const socketRef = useRef(null);
    const notifySocketRef = useRef(null);
    const notifyReconnectTimerRef = useRef(null);
    const isHistoryLoadingRef = useRef(false);
    const pendingMessagesRef = useRef([]);

    const currentRoomRef = useRef(currentRoom);
    const currentTabRef = useRef(currentTab);
    const currentUserRef = useRef(currentUser);
    const isOpenRef = useRef(isOpen);
    const friendsRef = useRef(friends);

    useEffect(() => {
        currentRoomRef.current = currentRoom;
    }, [currentRoom]);

    useEffect(() => {
        currentTabRef.current = currentTab;
    }, [currentTab]);

    useEffect(() => {
        currentUserRef.current = currentUser;
    }, [currentUser]);

    useEffect(() => {
        isOpenRef.current = isOpen;
    }, [isOpen]);

    useEffect(() => {
        friendsRef.current = friends;
    }, [friends]);

    const buildMessengerApiUrl = (path) => {
        const messengerPath = path.startsWith('/api/chat/')
            ? path.replace('/api/chat/', '/api/messenger/chat/')
            : path;
        return buildApiUrl(messengerPath);
    };

    const updateTotalUnreadFromAPI = async () => {
        try {
            const res = await fetch(buildMessengerApiUrl('/api/chat/rooms/'), { credentials: 'include' });
            if (!res.ok) return;
            const roomsData = await res.json();
            const isPopupOpen = isOpenRef.current;
            const currentRoomName = currentRoomRef.current?.name;
            const totalUnreadCount = roomsData.reduce((sum, room) => {
                if (isPopupOpen && room.name === currentRoomName) {
                    return sum;
                }
                return sum + (room.unread_count || 0);
            }, 0);
            setUnreadCount(totalUnreadCount);
        } catch (err) {
            console.error('읽지 않은 메시지 조회 실패:', err);
        }
    };

    const connectNotifications = () => {
        if (!currentUserRef.current) {
            console.log('알림 WebSocket: 사용자 정보 없음');
            return;
        }
        const existing = notifySocketRef.current;
        if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
            console.log('알림 WebSocket: 이미 연결 중');
            return;
        }

        const wsUrl = `${WS_BASE_URL}/ws/notifications/`;
        console.log('알림 WebSocket 연결 시도:', wsUrl);
        const socket = new WebSocket(wsUrl);
        notifySocketRef.current = socket;

        socket.onopen = () => {
            console.log('알림 WebSocket 연결 성공');
        };

        socket.onmessage = (event) => {
            let data = null;
            try {
                data = JSON.parse(event.data);
            } catch (err) {
                console.error('알림 메시지 파싱 실패:', err);
                return;
            }

            console.log('알림 수신:', data.type);

            if (data.type === 'notify_message') {
                updateTotalUnreadFromAPI();
                if (currentTabRef.current === 'chats') {
                    loadRooms();
                }
                return;
            }

            if (data.type === 'notify_read') {
                updateTotalUnreadFromAPI();
            }
        };

        socket.onerror = (error) => {
            console.error('알림 WebSocket 오류:', error);
        };

        socket.onclose = (event) => {
            console.log('알림 WebSocket 연결 종료:', event.code, event.reason);
            if (notifySocketRef.current === socket) {
                notifySocketRef.current = null;
            }
            if (notifyReconnectTimerRef.current) return;
            notifyReconnectTimerRef.current = setTimeout(() => {
                notifyReconnectTimerRef.current = null;
                console.log('알림 WebSocket 재연결 시도...');
                connectNotifications();
            }, 5000);
        };
    };
    const loadFriends = async () => {
        try {
            const res = await fetch(buildMessengerApiUrl('/api/chat/users/'), { credentials: 'include' });
            if (!res.ok) return;
            const users = await res.json();
            const me = currentUserRef.current;
            const list = me ? users.filter((u) => u.id !== me.id) : users;
            setFriends(list);
        } catch (err) {
            console.error('친구 로드 실패:', err);
        }
    };

    const loadRooms = async () => {
        try {
            const res = await fetch(buildMessengerApiUrl('/api/chat/rooms/'), { credentials: 'include' });
            if (!res.ok) return;
            const roomsData = await res.json();
            const me = currentUserRef.current;

            const totalUnreadCount = roomsData.reduce((sum, room) => sum + (room.unread_count || 0), 0);
            setUnreadCount(totalUnreadCount);

            const userMap = new Map();
            friendsRef.current.forEach((user) => userMap.set(user.id, user));
            if (me) userMap.set(me.id, me);

            const roomsWithNames = await Promise.all(
                roomsData.map(async (room) => {
                    let friendName = null;

                    if (room.room_type === 'group' && room.participants && me) {
                        const isMe = room.participants.some((u) => u.id === me.id);
                        if (!isMe) return null;
                    }

                    if (room.name.startsWith('case:dm:') && me) {
                        const parts = room.name.split(':');
                        const ids = parts.slice(2).map((id) => parseInt(id, 10));
                        const friendId = ids.find((id) => id !== me.id);

                        if (room.participants) {
                            const friend = room.participants.find((u) => u.id === friendId);
                            if (friend) friendName = friend.name || friend.username;
                        }

                        if (!friendName && friendId) {
                            const friend = userMap.get(friendId);
                            friendName = friend?.name || friend?.username || null;
                        }

                        if (!friendName) friendName = '1:1 대화';
                    } else if (room.name.startsWith('group:')) {
                        if (room.participants && me) {
                            const names = room.participants
                                .filter((u) => u.id !== me.id)
                                .map((u) => u.name || u.username);
                            const uniqueNames = [...new Set(names)];
                            friendName = uniqueNames.join(', ');
                        }
                        if (!friendName) friendName = '그룹 채팅';
                    }

                    return { ...room, friendName };
                })
            );

            setRooms(roomsWithNames.filter(Boolean));
        } catch (err) {
            console.error('대화방 로드 실패:', err);
        }
    };

    const loadData = () => {
        if (!currentUserRef.current) return;
        if (currentTabRef.current === 'friends') {
            loadFriends();
        } else {
            loadRooms();
        }
    };

    const updateHeaderForRoom = async (room) => {
        if (!room) {
            setHeaderTitle('의료진 채팅');
            setHeaderStatus('대기 중');
            return;
        }

        const me = currentUserRef.current;

        if (room.name && room.name.startsWith('case:dm:') && me) {
            let title = room.friendName || null;
            if (!title && room.participants) {
                const friend = room.participants.find((u) => u.id !== me.id);
                title = friend?.name || friend?.username || null;
            }
            if (!title) title = '1:1 대화';
            setHeaderTitle(title);
            return;
        }

        if (room.name && room.name.startsWith('group:')) {
            if (room.friendName) {
                setHeaderTitle(room.friendName);
            } else if (room.participants && me) {
                const names = room.participants
                    .filter((u) => u.id !== me.id)
                    .map((u) => u.name || u.username);
                const uniqueNames = [...new Set(names)];
                setHeaderTitle(uniqueNames.join(', ') || '그룹 채팅');
            } else {
                setHeaderTitle('그룹 채팅');
            }

            if (room.participant_count) {
                setHeaderStatus(`${room.participant_count}명`);
            } else if (room.participants) {
                setHeaderStatus(`${room.participants.length}명`);
            }
            return;
        }

        setHeaderTitle(getRoomLabel(room));
    };

    const markRoomAsRead = async (roomId) => {
        if (!roomId) return;
        try {
            await fetch(buildMessengerApiUrl(`/api/chat/rooms/${roomId}/mark-read/`), {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'Content-Type': 'application/json',
                },
            });

            setRooms((prev) => prev.map((room) => (
                room.id === roomId ? { ...room, unread_count: 0 } : room
            )));
            updateTotalUnreadFromAPI();
        } catch (err) {
            console.error('읽음 처리 실패:', err);
        }
    };

    const refreshReadStatusFromAPI = async (roomId) => {
        if (!roomId) return;
        try {
            const res = await fetch(buildMessengerApiUrl(`/api/chat/rooms/${roomId}/messages/?limit=50&mark_read=0`), {
                credentials: 'include',
            });
            if (!res.ok) return;
            const messagesData = await res.json();
            const messageMap = new Map(messagesData.map((msg) => [String(msg.id), msg]));

            setMessages((prev) => prev.map((msg) => {
                const updated = messageMap.get(String(msg.id));
                if (!updated) return msg;
                return { ...msg, unread_count: updated.unread_count };
            }));
        } catch (err) {
            console.error('읽음 상태 갱신 실패:', err);
        }
    };

    const getRoomIdByName = async (roomName) => {
        try {
            const res = await fetch(buildMessengerApiUrl('/api/chat/rooms/'), { credentials: 'include' });
            if (!res.ok) return null;
            const roomsData = await res.json();
            const found = roomsData.find((room) => room.name === roomName);
            return found ? found.id : null;
        } catch (err) {
            return null;
        }
    };

    const connectToRoom = async (roomName, roomId, roomData) => {
        if (!roomName) return;

        if (socketRef.current) {
            socketRef.current.onclose = null;
            socketRef.current.close();
            socketRef.current = null;
        }

        const nextRoom = roomData || { id: roomId, name: roomName };
        setCurrentRoom(nextRoom);
        setMessages([]);
        pendingMessagesRef.current = [];
        isHistoryLoadingRef.current = false;

        await updateHeaderForRoom(nextRoom);

        if (roomId) {
            markRoomAsRead(roomId);
        }
    };

    const openDM = async (targetUserId) => {
        const me = currentUserRef.current;
        if (!me) return;

        try {
            const u1 = Math.min(me.id, targetUserId);
            const u2 = Math.max(me.id, targetUserId);
            const caseKey = `dm:${u1}:${u2}`;

            const res = await fetch(buildMessengerApiUrl('/api/chat/rooms/'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken'),
                },
                body: JSON.stringify({
                    room_type: 'case',
                    case_key: caseKey,
                    participant_ids: [targetUserId],
                }),
                credentials: 'include',
            });

            if (!res.ok) {
                const txt = await res.text();
                console.error('DM Create Error:', txt);
                alert('대화방 열기 실패');
                return;
            }

            const room = await res.json();
            await connectToRoom(room.name, room.id, room);
        } catch (err) {
            console.error('DM Error:', err);
            alert('오류가 발생했습니다.');
        }
    };

    const createGroupChat = async () => {
        if (selectedUserIds.size === 0) return;

        try {
            const userIds = Array.from(selectedUserIds);
            const res = await fetch(buildMessengerApiUrl('/api/chat/rooms/'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken'),
                },
                body: JSON.stringify({
                    room_type: 'group',
                    participant_ids: userIds,
                }),
                credentials: 'include',
            });

            if (!res.ok) {
                console.error('그룹 생성 실패', await res.text());
                alert('그룹 생성 실패');
                return;
            }

            const room = await res.json();
            setIsSelectionMode(false);
            setSelectedUserIds(new Set());
            await connectToRoom(room.name, room.id, room);
        } catch (err) {
            console.error('에러:', err);
        }
    };

    const showListView = () => {
        setCurrentRoom(null);
        setMessages([]);
        setHeaderTitle('의료진 채팅');
        setHeaderStatus('대기 중');
        if (socketRef.current) {
            socketRef.current.onclose = null;
            socketRef.current.close();
            socketRef.current = null;
        }
        loadData();
    };

    const leaveRoom = async () => {
        const roomId = currentRoomRef.current?.id;
        if (!roomId) return;
        if (!confirm('채팅방을 나가시겠습니까? 대화방 목록에서 사라집니다.')) return;

        try {
            const res = await fetch(buildMessengerApiUrl(`/api/chat/rooms/${roomId}/leave/`), {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                },
                credentials: 'include',
            });

            if (res.ok || res.status === 204) {
                setCurrentTab('chats');
                showListView();
                updateTotalUnreadFromAPI();
            } else {
                alert(`나가기 실패 (코드: ${res.status})`);
            }
        } catch (err) {
            alert(`오류 발생: ${err.message}`);
        }
    };

    const sendMessage = () => {
        const content = messageInput.trim();
        if (!content) return;
        const socket = socketRef.current;
        
        if (!socket) {
            console.error('WebSocket이 연결되지 않았습니다.');
            alert('채팅 연결이 끊어졌습니다. 페이지를 새로고침해주세요.');
            return;
        }
        
        if (socket.readyState !== WebSocket.OPEN) {
            console.error('WebSocket 상태:', socket.readyState);
            alert('채팅 연결이 준비되지 않았습니다. 잠시 후 다시 시도해주세요.');
            return;
        }

        try {
            const payload = {
                type: 'chat_message',
                content,
            };
            console.log('메시지 전송:', payload);
            socket.send(JSON.stringify(payload));

            setMessageInput('');
            if (messageInputRef.current) {
                messageInputRef.current.style.height = 'auto';
            }
        } catch (err) {
            console.error('메시지 전송 실패:', err);
            alert('메시지 전송에 실패했습니다. 다시 시도해주세요.');
        }
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const me = currentUserRef.current;
        if (!me) {
            alert('사용자 정보를 불러오는 중입니다. 잠시 후 다시 시도해주세요.');
            event.target.value = '';
            return;
        }

        const roomName = currentRoomRef.current?.name;
        const roomId = currentRoomRef.current?.id || (roomName ? await getRoomIdByName(roomName) : null);
        if (!roomId) {
            alert('채팅방을 찾을 수 없습니다.');
            event.target.value = '';
            return;
        }

        try {
            const formData = new FormData();
            formData.append('file', file);

            const uploadRes = await fetch(buildMessengerApiUrl('/api/chat/files/'), {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                },
                body: formData,
            });

            if (!uploadRes.ok) {
                const errorText = await uploadRes.text();
                try {
                    const errorData = JSON.parse(errorText);
                    if (errorData.error && errorData.error[0] === 'File extension not allowed.') {
                        alert(
                            '⚠️ 지원하지 않는 파일 형식입니다.\n\n' +
                            '파일 전송은 다음 형식만 가능합니다:\n\n' +
                            '이미지: PNG, JPG, JPEG, GIF, WEBP\n' +
                            '문서: PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT\n' +
                            '텍스트: TXT, MD\n' +
                            '한글: HWP, HWPX'
                        );
                    } else if (errorData.error && errorData.error[0].includes('exceeds')) {
                        alert('⚠️ 파일 크기가 너무 큽니다.\n\n최대 50MB까지 업로드 가능합니다.');
                    } else {
                        alert(`파일 업로드 실패: ${errorData.error?.[0] || errorText}`);
                    }
                } catch (err) {
                    alert(`파일 업로드 실패: ${errorText}`);
                }
                event.target.value = '';
                return;
            }

            const uploadData = await uploadRes.json();
            const msgRes = await fetch(buildMessengerApiUrl(`/api/chat/rooms/${roomId}/messages/`), {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: '',
                    attachment_id: uploadData.id,
                }),
            });

            if (!msgRes.ok) {
                const errorText = await msgRes.text();
                throw new Error(errorText || '파일 메시지 전송 실패');
            }
        } catch (err) {
            console.error('파일 전송 오류:', err);
            alert(`파일 전송 실패: ${err.message}`);
        }

        event.target.value = '';
    };

    const toggleSelectionMode = () => {
        setIsSelectionMode(true);
        setSelectedUserIds(new Set());
        loadFriends();
    };

    const cancelSelectionMode = () => {
        setIsSelectionMode(false);
        setSelectedUserIds(new Set());
        loadFriends();
    };

    const toggleUserSelection = (userId) => {
        setSelectedUserIds((prev) => {
            const next = new Set(prev);
            if (next.has(userId)) {
                next.delete(userId);
            } else {
                next.add(userId);
            }
            return next;
        });
    };

    useEffect(() => {
        const init = async () => {
            try {
                const res = await fetch(buildMessengerApiUrl('/api/chat/users/me/'), { credentials: 'include' });
                if (!res.ok) return;
                const user = await res.json();
                currentUserRef.current = user;
                setCurrentUser(user);
                connectNotifications();
                updateTotalUnreadFromAPI();
            } catch (err) {
                console.error('사용자 정보 로드 실패:', err);
            }
        };

        init();
    }, []);

    useEffect(() => {
        if (isOpen && currentUser) {
            loadData();
        }
    }, [isOpen, currentTab, currentUser]);

    useEffect(() => {
        const interval = setInterval(() => {
            if (!isOpenRef.current) {
                updateTotalUnreadFromAPI();
            }
        }, 30000);

        return () => clearInterval(interval);
    }, []);

    const loadMessagesFromAPI = async (roomId) => {
        isHistoryLoadingRef.current = true;
        pendingMessagesRef.current = [];

        try {
            const res = await fetch(buildMessengerApiUrl(`/api/chat/rooms/${roomId}/messages/?limit=50&mark_read=1`), {
                credentials: 'include',
            });
            if (!res.ok) return;
            const messagesData = await res.json();
            const baseMessages = messagesData || [];
            const existingIds = new Set(baseMessages.map((msg) => msg.id));
            const pendingMessages = pendingMessagesRef.current.filter((msg) => !existingIds.has(msg.id));
            setMessages([...baseMessages, ...pendingMessages]);
        } catch (err) {
            console.error('메시지 로드 실패:', err);
        } finally {
            isHistoryLoadingRef.current = false;
            pendingMessagesRef.current = [];
        }
    };

    useEffect(() => {
        if (!currentRoom?.name) return;

        const wsUrl = `${WS_BASE_URL}/ws/chat/${currentRoom.name}/`;
        console.log('WebSocket 연결 시도:', wsUrl, 'WS_BASE_URL:', WS_BASE_URL);
        const socket = new WebSocket(wsUrl);
        socketRef.current = socket;

        setHeaderStatus('연결 중...');

        socket.onopen = () => {
            console.log('WebSocket 연결 성공:', wsUrl);
            setHeaderStatus('연결됨');
        };

        socket.onmessage = (event) => {
            let data = null;
            try {
                data = JSON.parse(event.data);
            } catch (err) {
                return;
            }

            if (data.type === 'message_history') {
                const roomId = currentRoomRef.current?.id;
                if (roomId) {
                    loadMessagesFromAPI(roomId);
                } else if (data.data?.messages) {
                    setMessages(data.data.messages || []);
                }
                return;
            }

            if (data.type === 'chat_message') {
                const message = data.data.message;
                const me = currentUserRef.current;
                if (me && message?.sender?.id !== me.id) {
                    setTimeout(() => {
                        const roomId = currentRoomRef.current?.id;
                        if (roomId) markRoomAsRead(roomId);
                    }, 500);
                }

                if (isHistoryLoadingRef.current) {
                    pendingMessagesRef.current.push(message);
                } else {
                    setMessages((prev) => {
                        if (prev.some((msg) => msg.id === message.id)) return prev;
                        return [...prev, message];
                    });
                }
                return;
            }

            if (data.type === 'message_read_status') {
                const readUserId = data.data.user_id;
                const isDMRoom = currentRoomRef.current?.name?.startsWith('case:dm:');
                const me = currentUserRef.current;
                if (!isDMRoom || readUserId !== me?.id) {
                    refreshReadStatusFromAPI(currentRoomRef.current?.id);
                }
            }
        };

        socket.onerror = (error) => {
            console.error('WebSocket 오류:', error, 'URL:', wsUrl);
            setHeaderStatus('연결 오류');
        };

        socket.onclose = (event) => {
            console.log('WebSocket 연결 종료:', event.code, event.reason, 'wasClean:', event.wasClean, 'URL:', wsUrl);
            // cleanup에서 호출된 경우가 아니고, 현재 방이 여전히 같은 경우에만 상태 업데이트
            if (socketRef.current === socket && currentRoomRef.current?.name === currentRoom.name) {
                if (event.code !== 1000) {  // 정상 종료가 아닌 경우
                    setHeaderStatus('연결 종료');
                    // 비정상 종료 시 재연결 시도
                    const roomName = currentRoomRef.current.name;
                    setTimeout(() => {
                        if (currentRoomRef.current?.name === roomName && !socketRef.current) {
                            console.log('WebSocket 재연결 시도...');
                            // 재연결을 위해 currentRoom을 다시 설정
                            const room = currentRoomRef.current;
                            setCurrentRoom(null);
                            setTimeout(() => {
                                setCurrentRoom(room);
                            }, 100);
                        }
                    }, 3000);
                }
            }
        };

        return () => {
            socket.onclose = null;
            socket.close();
            if (socketRef.current === socket) {
                socketRef.current = null;
            }
        };
    }, [currentRoom?.name]);

    useEffect(() => {
        const container = chatMessagesRef.current;
        if (!container) return;
        container.scrollTop = container.scrollHeight;
    }, [messages]);

    useEffect(() => {
        const handleFocus = () => {
            const roomId = currentRoomRef.current?.id;
            if (roomId) markRoomAsRead(roomId);
        };

        window.addEventListener('focus', handleFocus);
        return () => window.removeEventListener('focus', handleFocus);
    }, []);

    const renderedMessages = useMemo(() => {
        const items = [];
        let lastDate = null;
        const me = currentUser;
        const isDMRoom = currentRoom?.name?.startsWith('case:dm:');

        messages.forEach((msg) => {
            const dateLabel = formatDateLabel(msg.timestamp);
            if (dateLabel && lastDate !== dateLabel) {
                lastDate = dateLabel;
                items.push(
                    <div className="chat-date-divider" key={`date-${dateLabel}`}>
                        <span className="chat-date-label">{dateLabel}</span>
                    </div>
                );
            }

            const sender = msg.sender || {};
            const senderName = sender.name || sender.username || '알 수 없음';
            const isMine = me && sender.id === me.id;
            let displayUnreadCount = typeof msg.unread_count === 'number' ? msg.unread_count : 0;
            if (isDMRoom && !isMine) {
                displayUnreadCount = 0;
            }

            items.push(
                <div className={`chat-message ${isMine ? 'mine' : ''}`} key={msg.id}>
                    {!isMine && (
                        <div className="chat-message-avatar">{senderName[0] || '?'}</div>
                    )}
                    <div className="chat-message-content">
                        {!isMine && (
                            <div className="chat-message-name">{senderName}</div>
                        )}
                        <div className="message-body">
                            <div className="message-bubble-wrapper">
                                {msg.content && (
                                    <div className="chat-message-bubble">{msg.content}</div>
                                )}
                                {msg.attachment_url && (
                                    <div className="chat-message-attachment">
                                        <a href={msg.attachment_url} target="_blank" rel="noreferrer">
                                            {isImage(msg.attachment_name) ? (
                                                <img src={msg.attachment_url} alt={msg.attachment_name || ''} />
                                            ) : (
                                                `📎 ${msg.attachment_name || '첨부파일'}`
                                            )}
                                        </a>
                                    </div>
                                )}
                            </div>
                            <div className="chat-message-meta">
                                {displayUnreadCount > 0 && (
                                    <span className="read-status unread">{displayUnreadCount}</span>
                                )}
                                <span className="chat-message-time">{formatTime(msg.timestamp)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            );
        });

        return items;
    }, [messages, currentRoom, currentUser]);

    const unreadBadge = unreadCount > 0 ? (unreadCount > 99 ? '99+' : unreadCount) : null;
    return (
        <div className="floating-chat">
            {!isOpen && (
                <button
                    className="chat-trigger"
                    onClick={() => setIsOpen(true)}
                    title="채팅 열기"
                    type="button"
                >
                    <svg viewBox="0 0 24 24">
                        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
                    </svg>
                    {unreadBadge && (
                        <span className="badge">{unreadBadge}</span>
                    )}
                </button>
            )}

            <div className={`chat-popup ${isOpen ? 'active' : ''}`}>
                <div className="chat-header">
                    <div className="chat-header-left">
                        <div>
                            <div className="chat-header-title">{headerTitle}</div>
                            <div className="chat-header-status">{headerStatus}</div>
                        </div>
                    </div>
                    {currentRoom && (
                        <button
                            className="chat-header-btn"
                            onClick={showListView}
                            title="돌아가기"
                            type="button"
                        >
                            <svg viewBox="0 0 24 24">
                                <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
                            </svg>
                        </button>
                    )}
                    {currentRoom && (
                        <button
                            className="chat-header-btn"
                            onClick={leaveRoom}
                            title="나가기"
                            type="button"
                        >
                            <svg viewBox="0 0 24 24" style={{ width: '20px', height: '20px' }}>
                                <path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z" />
                            </svg>
                        </button>
                    )}
                    <button
                        className="chat-header-btn"
                        onClick={() => setIsOpen(false)}
                        title="최소화"
                        type="button"
                    >
                        <svg viewBox="0 0 24 24">
                            <path d="M19 13H5v-2h14v2z" />
                        </svg>
                    </button>
                </div>

                <div className={`chat-view ${currentRoom ? 'hidden' : ''}`}>
                    <div className="chat-tabs">
                        <button
                            className={`chat-tab ${currentTab === 'friends' ? 'active' : ''}`}
                            onClick={() => setCurrentTab('friends')}
                            type="button"
                        >
                            동료
                        </button>
                        <button
                            className={`chat-tab ${currentTab === 'chats' ? 'active' : ''}`}
                            onClick={() => setCurrentTab('chats')}
                            type="button"
                        >
                            대화방
                        </button>
                    </div>

                    {currentTab === 'friends' && (
                        <div>
                            {!isSelectionMode && (
                                <div className="group-start-btn-area">
                                    <button className="group-start-btn" onClick={toggleSelectionMode} type="button">
                                        + 그룹채팅
                                    </button>
                                </div>
                            )}
                            {isSelectionMode && (
                                <div className="group-control-bar">
                                    <strong>{selectedUserIds.size}명 선택됨</strong>
                                    <div style={{ display: 'flex', gap: '5px' }}>
                                        <button
                                            className="group-start-btn"
                                            onClick={createGroupChat}
                                            disabled={selectedUserIds.size === 0}
                                            style={{
                                                color: 'var(--primary)',
                                                fontWeight: 'bold',
                                                borderColor: 'var(--primary)',
                                                opacity: selectedUserIds.size > 0 ? 1 : 0.5,
                                            }}
                                            type="button"
                                        >
                                            확인
                                        </button>
                                        <button className="group-start-btn" onClick={cancelSelectionMode} type="button">
                                            취소
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="chat-list">
                        {currentTab === 'friends' && friends.length === 0 && (
                            <div className="chat-empty">
                                <svg viewBox="0 0 24 24">
                                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
                                </svg>
                                <h3>등록된 동료가 없습니다</h3>
                                <p>동료를 추가하고 대화를 시작하세요</p>
                            </div>
                        )}

                        {currentTab === 'friends' && friends.map((user) => (
                            <div
                                key={user.id}
                                className={`chat-list-item ${isSelectionMode ? 'selection-mode' : ''}`}
                                onClick={() => {
                                    if (isSelectionMode) {
                                        toggleUserSelection(user.id);
                                    } else {
                                        openDM(user.id);
                                    }
                                }}
                            >
                                {isSelectionMode && (
                                    <input
                                        type="checkbox"
                                        className="friend-checkbox"
                                        checked={selectedUserIds.has(user.id)}
                                        onChange={() => toggleUserSelection(user.id)}
                                        onClick={(event) => event.stopPropagation()}
                                    />
                                )}
                                <div className={`chat-list-avatar ${user.is_online ? 'online' : ''}`}>
                                    {(user.name || user.username || '?')[0]}
                                </div>
                                <div className="chat-list-content">
                                    <div className="chat-list-top">
                                        <span className="chat-list-name">{user.name || user.username}</span>
                                        <span className="chat-list-time">{user.is_online ? '온라인' : ''}</span>
                                    </div>
                                    <div className="chat-list-preview">{user.department || user.role || '의료진'}</div>
                                </div>
                            </div>
                        ))}

                        {currentTab === 'chats' && rooms.length === 0 && (
                            <div className="chat-empty">
                                <svg viewBox="0 0 24 24">
                                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
                                </svg>
                                <h3>대화방이 없습니다</h3>
                                <p>동료를 선택하여 채팅을 시작하세요</p>
                            </div>
                        )}

                        {currentTab === 'chats' && rooms.map((room) => (
                            <div
                                key={room.id}
                                className="chat-list-item"
                                onClick={() => connectToRoom(room.name, room.id, room)}
                            >
                                <div className="chat-list-avatar">💬</div>
                                <div className="chat-list-content">
                                    <div className="chat-list-top">
                                        <span className="chat-list-name">{room.friendName || getRoomLabel(room)}</span>
                                        <span className="chat-list-time">{formatTime(room.last_message_at || room.created_at)}</span>
                                    </div>
                                    <div className="chat-list-preview">{room.last_message?.content || '새 대화'}</div>
                                </div>
                                {room.unread_count > 0 && (
                                    <span className="chat-list-badge">{room.unread_count > 99 ? '99+' : room.unread_count}</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                <div className={`chat-room ${currentRoom ? '' : 'hidden'}`}>
                    <div
                        className="chat-messages"
                        ref={chatMessagesRef}
                        onClick={() => {
                            const roomId = currentRoomRef.current?.id;
                            if (roomId) markRoomAsRead(roomId);
                        }}
                    >
                        {renderedMessages}
                    </div>
                    <div className="chat-composer">
                        <button
                            className="chat-composer-btn"
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            title="파일 첨부"
                        >
                            <svg viewBox="0 0 24 24">
                                <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" />
                            </svg>
                        </button>
                        <input
                            type="file"
                            ref={fileInputRef}
                            style={{ display: 'none' }}
                            onChange={handleFileUpload}
                        />
                        <textarea
                            className="chat-composer-input"
                            ref={messageInputRef}
                            value={messageInput}
                            placeholder="메시지를 입력하세요"
                            rows={1}
                            onChange={(event) => {
                                setMessageInput(event.target.value);
                                event.target.style.height = 'auto';
                                event.target.style.height = `${event.target.scrollHeight}px`;
                            }}
                            onFocus={() => {
                                const roomId = currentRoomRef.current?.id;
                                if (roomId) markRoomAsRead(roomId);
                            }}
                            onKeyDown={(event) => {
                                if (event.key === 'Enter' && !event.shiftKey) {
                                    event.preventDefault();
                                    sendMessage();
                                }
                            }}
                        />
                        <button
                            className="chat-composer-send"
                            type="button"
                            onClick={sendMessage}
                            disabled={!messageInput.trim()}
                        >
                            <svg viewBox="0 0 24 24">
                                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FloatingChat;
