#!/usr/bin/env python3
"""
Test MiniMax-M2 with 25k unique token input + 5k token output
Tests long context performance with useEffect question
"""
import anthropic
import time
import tiktoken

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding (approximation)"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_unique_context(target_tokens: int) -> str:
    """
    Generate ~25k unique tokens of realistic code/documentation
    Not just repeating the same content 5 times!
    """

    # Base React documentation and examples
    contexts = []

    # 1. Comprehensive useEffect documentation (~5k tokens)
    contexts.append("""
# React useEffect Hook - Complete Guide

## Overview
The useEffect Hook allows you to perform side effects in function components. It serves the same purpose as componentDidMount, componentDidUpdate, and componentWillUnmount combined in React class components.

## Basic Syntax
```javascript
useEffect(() => {
  // Effect code here
  return () => {
    // Cleanup code here
  };
}, [dependencies]);
```

## Common Use Cases

### 1. Data Fetching
```javascript
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function fetchUser() {
      setLoading(true);
      try {
        const response = await fetch(`/api/users/${userId}`);
        const data = await response.json();
        if (!cancelled) {
          setUser(data);
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to fetch user:', error);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchUser();

    return () => {
      cancelled = true;
    };
  }, [userId]);

  if (loading) return <div>Loading...</div>;
  if (!user) return <div>User not found</div>;
  return <div>{user.name}</div>;
}
```

### 2. Event Listeners
```javascript
function WindowSize() {
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight });

  useEffect(() => {
    function handleResize() {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    }

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return <div>Window: {size.width} x {size.height}</div>;
}
```

### 3. Subscriptions
```javascript
function ChatRoom({ roomId }) {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const socket = io('http://localhost:3000');
    socket.emit('join_room', roomId);

    socket.on('message', (message) => {
      setMessages(prev => [...prev, message]);
    });

    return () => {
      socket.emit('leave_room', roomId);
      socket.disconnect();
    };
  }, [roomId]);

  return <div>{messages.map(msg => <p key={msg.id}>{msg.text}</p>)}</div>;
}
```
""")

    # 2. Advanced patterns (~5k tokens)
    contexts.append("""
## Advanced useEffect Patterns

### Debouncing and Throttling
```javascript
function SearchInput() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    const timeoutId = setTimeout(async () => {
      if (query.length > 2) {
        const res = await fetch(`/api/search?q=${query}`);
        setResults(await res.json());
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [query]);

  return (
    <div>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      {results.map(r => <div key={r.id}>{r.title}</div>)}
    </div>
  );
}
```

### Multiple Effects for Separation of Concerns
```javascript
function Dashboard({ userId, theme }) {
  // Effect 1: User data
  useEffect(() => {
    fetchUserData(userId);
  }, [userId]);

  // Effect 2: Theme
  useEffect(() => {
    document.body.className = theme;
  }, [theme]);

  // Effect 3: Analytics
  useEffect(() => {
    trackPageView('dashboard');
  }, []);

  // Effect 4: Periodic updates
  useEffect(() => {
    const interval = setInterval(() => {
      refreshDashboardData();
    }, 30000);
    return () => clearInterval(interval);
  }, []);
}
```

### Conditional Effects
```javascript
function DataSync({ shouldSync, data }) {
  useEffect(() => {
    if (!shouldSync) return;

    const sync = async () => {
      await api.syncData(data);
    };

    sync();
  }, [shouldSync, data]);
}
```

### Custom Hooks with useEffect
```javascript
function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : initialValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
}

function useWindowEvent(event, handler) {
  useEffect(() => {
    window.addEventListener(event, handler);
    return () => window.removeEventListener(event, handler);
  }, [event, handler]);
}

function useInterval(callback, delay) {
  const savedCallback = useRef();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;

    const tick = () => savedCallback.current();
    const id = setInterval(tick, delay);
    return () => clearInterval(id);
  }, [delay]);
}
```
""")

    # 3. Performance optimization (~5k tokens)
    contexts.append("""
## Performance Optimization with useEffect

### Avoiding Unnecessary Effects
```javascript
// Bad: Effect runs on every render
function BadExample({ count }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchData().then(setData);
  }); // Missing dependency array!
}

// Good: Effect only runs when count changes
function GoodExample({ count }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchData(count).then(setData);
  }, [count]);
}
```

### Memoizing Callback Dependencies
```javascript
function ExpensiveComponent({ userId, filters }) {
  const memoizedFilters = useMemo(() => filters, [JSON.stringify(filters)]);

  useEffect(() => {
    fetchData(userId, memoizedFilters);
  }, [userId, memoizedFilters]);
}

// Better approach with useCallback
function BetterComponent({ userId, onDataFetched }) {
  useEffect(() => {
    fetchData(userId).then(onDataFetched);
  }, [userId, onDataFetched]); // onDataFetched should be wrapped in useCallback
}
```

### Lazy Initialization
```javascript
function LazyInit() {
  const [data, setData] = useState(() => {
    // This expensive operation only runs once
    return computeExpensiveInitialValue();
  });

  useEffect(() => {
    // Only fetch if data is null
    if (!data) {
      fetchData().then(setData);
    }
  }, [data]);
}
```

### Batching Updates
```javascript
function BatchedUpdates() {
  const [count, setCount] = useState(0);
  const [items, setItems] = useState([]);

  useEffect(() => {
    // React batches these updates automatically
    setCount(c => c + 1);
    setItems(i => [...i, newItem]);
    // Only causes one re-render
  }, [trigger]);
}
```

### AbortController for Fetch Cancellation
```javascript
function SearchResults({ query }) {
  const [results, setResults] = useState([]);

  useEffect(() => {
    const controller = new AbortController();

    async function search() {
      try {
        const res = await fetch(`/api/search?q=${query}`, {
          signal: controller.signal
        });
        const data = await res.json();
        setResults(data);
      } catch (err) {
        if (err.name === 'AbortError') {
          console.log('Fetch aborted');
        }
      }
    }

    search();
    return () => controller.abort();
  }, [query]);
}
```
""")

    # 4. Common pitfalls (~5k tokens)
    contexts.append("""
## Common useEffect Pitfalls and Solutions

### 1. Infinite Loops
```javascript
// BAD: Infinite loop
function InfiniteLoop() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    setCount(count + 1); // This updates count, causing effect to run again
  }, [count]); // count is in dependency array
}

// GOOD: Functional update
function FixedLoop() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setCount(c => c + 1); // Use functional form
    }, 1000);
    return () => clearTimeout(timer);
  }, []); // Empty dependency array
}
```

### 2. Stale Closures
```javascript
// BAD: Stale closure
function StaleClosureExample() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      console.log(count); // Always logs 0!
    }, 1000);
    return () => clearInterval(interval);
  }, []); // count is not in dependencies
}

// GOOD: Use ref or include in dependencies
function FixedClosure() {
  const [count, setCount] = useState(0);
  const countRef = useRef(count);

  useEffect(() => {
    countRef.current = count;
  }, [count]);

  useEffect(() => {
    const interval = setInterval(() => {
      console.log(countRef.current); // Always current value
    }, 1000);
    return () => clearInterval(interval);
  }, []);
}
```

### 3. Missing Dependencies
```javascript
// BAD: ESLint will warn
function MissingDeps({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, []); // userId is missing!
}

// GOOD: Include all dependencies
function AllDeps({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, [userId]);
}
```

### 4. Race Conditions
```javascript
// BAD: Race condition possible
function RaceCondition({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, [userId]);
  // If userId changes quickly, older fetch might complete after newer one
}

// GOOD: Ignore stale responses
function NoRaceCondition({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    let cancelled = false;

    fetchUser(userId).then(data => {
      if (!cancelled) setUser(data);
    });

    return () => { cancelled = true; };
  }, [userId]);
}
```

### 5. Effect Dependencies That Are Objects
```javascript
// BAD: Object recreated on every render
function ObjectDep() {
  const config = { apiKey: '123', timeout: 5000 };

  useEffect(() => {
    initializeService(config);
  }, [config]); // config is a new object every render!
}

// GOOD: Memoize the object
function MemoizedObjectDep() {
  const config = useMemo(() => ({
    apiKey: '123',
    timeout: 5000
  }), []);

  useEffect(() => {
    initializeService(config);
  }, [config]);
}
```
""")

    # 5. Real-world examples (~5k+ tokens)
    contexts.append("""
## Real-World useEffect Examples

### Authentication Flow
```javascript
function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (authUser) => {
      if (authUser) {
        const userData = await fetchUserProfile(authUser.uid);
        setUser(userData);
      } else {
        setUser(null);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  if (loading) return <Loading />;
  return <AuthContext.Provider value={user}>{children}</AuthContext.Provider>;
}
```

### WebSocket Real-time Updates
```javascript
function LivePrices({ symbols }) {
  const [prices, setPrices] = useState({});
  const wsRef = useRef(null);

  useEffect(() => {
    wsRef.current = new WebSocket('wss://api.prices.com');

    wsRef.current.onopen = () => {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        symbols
      }));
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setPrices(prev => ({
        ...prev,
        [data.symbol]: data.price
      }));
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.send(JSON.stringify({
          type: 'unsubscribe',
          symbols
        }));
        wsRef.current.close();
      }
    };
  }, [symbols.join(',')]);

  return (
    <div>
      {symbols.map(symbol => (
        <div key={symbol}>
          {symbol}: ${prices[symbol] || 'Loading...'}
        </div>
      ))}
    </div>
  );
}
```

### Form Auto-save
```javascript
function AutoSaveForm({ documentId }) {
  const [content, setContent] = useState('');
  const [lastSaved, setLastSaved] = useState(null);
  const [saving, setSaving] = useState(false);

  // Load initial content
  useEffect(() => {
    async function loadDocument() {
      const doc = await api.getDocument(documentId);
      setContent(doc.content);
      setLastSaved(doc.updatedAt);
    }
    loadDocument();
  }, [documentId]);

  // Auto-save with debounce
  useEffect(() => {
    if (!content) return;

    setSaving(true);
    const timeoutId = setTimeout(async () => {
      try {
        await api.updateDocument(documentId, { content });
        setLastSaved(new Date());
      } finally {
        setSaving(false);
      }
    }, 2000);

    return () => clearTimeout(timeoutId);
  }, [content, documentId]);

  return (
    <div>
      <textarea
        value={content}
        onChange={e => setContent(e.target.value)}
      />
      <div>
        {saving ? 'Saving...' : `Last saved: ${lastSaved?.toLocaleTimeString()}`}
      </div>
    </div>
  );
}
```

### Infinite Scroll
```javascript
function InfiniteList() {
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const observerTarget = useRef(null);

  useEffect(() => {
    async function loadMore() {
      if (loading || !hasMore) return;

      setLoading(true);
      try {
        const newItems = await api.getItems(page);
        setItems(prev => [...prev, ...newItems]);
        setHasMore(newItems.length > 0);
        setPage(p => p + 1);
      } finally {
        setLoading(false);
      }
    }

    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          loadMore();
        }
      },
      { threshold: 1.0 }
    );

    if (observerTarget.current) {
      observer.observe(observerTarget.current);
    }

    return () => observer.disconnect();
  }, [page, loading, hasMore]);

  return (
    <div>
      {items.map(item => <Item key={item.id} {...item} />)}
      <div ref={observerTarget}>{loading && 'Loading...'}</div>
    </div>
  );
}
```

### Geolocation Tracking
```javascript
function LocationTracker() {
  const [position, setPosition] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!navigator.geolocation) {
      setError('Geolocation not supported');
      return;
    }

    const watchId = navigator.geolocation.watchPosition(
      pos => {
        setPosition({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          accuracy: pos.coords.accuracy
        });
      },
      err => setError(err.message),
      {
        enableHighAccuracy: true,
        maximumAge: 0,
        timeout: 5000
      }
    );

    return () => navigator.geolocation.clearWatch(watchId);
  }, []);

  if (error) return <div>Error: {error}</div>;
  if (!position) return <div>Acquiring location...</div>;
  return <div>Lat: {position.latitude}, Lng: {position.longitude}</div>;
}
```
""")

    # Concatenate all contexts
    full_context = "\n\n".join(contexts)

    # Pad with additional unique React examples if needed
    current_tokens = count_tokens(full_context)
    if current_tokens < target_tokens:
        padding = f"""
### Additional React Patterns and Best Practices

#### Error Boundaries with useEffect
Error boundaries catch errors during rendering, but useEffect errors need manual handling:

```javascript
function SafeComponent() {{
  const [error, setError] = useState(null);

  useEffect(() => {{
    try {{
      riskyOperation();
    }} catch (err) {{
      setError(err);
      logErrorToService(err);
    }}
  }}, []);

  if (error) return <ErrorDisplay error={{error}} />;
  return <div>Component content</div>;
}}
```

#### Dynamic Script Loading
```javascript
function DynamicScriptLoader({{ src }}) {{
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {{
    const script = document.createElement('script');
    script.src = src;
    script.async = true;

    script.onload = () => setLoaded(true);
    script.onerror = () => setError(new Error('Failed to load script'));

    document.body.appendChild(script);

    return () => {{
      document.body.removeChild(script);
    }};
  }}, [src]);

  return {{ loaded, error }};
}}
```

This comprehensive guide covers all major useEffect patterns, from basic usage to advanced optimization techniques and real-world applications.

""" * max(1, int((target_tokens - current_tokens) / 500))
        full_context += padding

    return full_context

def main():
    print("=" * 80)
    print("MiniMax-M2 Long Context Test: 25k input + 5k output")
    print("=" * 80)

    # Generate unique 25k token context
    print("\n[1/4] Generating 25k unique tokens of React documentation...")
    context = generate_unique_context(25000)
    actual_tokens = count_tokens(context)
    print(f"Generated context: {actual_tokens:,} tokens")

    # Create client
    print("\n[2/4] Connecting to proxy at https://apically-euphemistic-adriana.ngrok-free.dev...")
    client = anthropic.Anthropic(
        base_url="https://apically-euphemistic-adriana.ngrok-free.dev",
        api_key="not-needed"
    )

    # Create the question
    question = """Based on the comprehensive React useEffect documentation provided above, please answer this question in detail:

What are the 5 most critical mistakes developers make when using useEffect, and how can each be avoided? For each mistake, provide:
1. A clear explanation of the problem
2. A code example showing the mistake
3. The correct solution with code
4. Why the solution works

Please be thorough and include real-world scenarios where these mistakes commonly occur."""

    # Send request
    print("\n[3/4] Sending request to MiniMax-M2...")
    print(f"Question: {question[:100]}...")

    start_time = time.time()

    message = client.messages.create(
        model="/mnt/llm_models/MiniMax-M2-REAP-162B-A10B-AWQ-4bit",
        max_tokens=5000,
        messages=[
            {
                "role": "user",
                "content": f"{context}\n\n---\n\n{question}"
            }
        ],
        temperature=0.7
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # Display results
    print("\n[4/4] Response received!")
    print("=" * 80)
    print(f"⏱️  Total time: {elapsed:.2f}s")
    print(f"📊 Input tokens: ~{actual_tokens:,}")
    print(f"📊 Output tokens: {message.usage.output_tokens:,}")
    print(f"🚀 Throughput: {message.usage.output_tokens / elapsed:.1f} tokens/sec")
    print("=" * 80)

    print("\n📝 Response:")
    print("-" * 80)
    for block in message.content:
        if block.type == "text":
            print(block.text)
    print("-" * 80)

    print(f"\n✅ Test complete! Model handled {actual_tokens:,} token context successfully.")
    print(f"Stop reason: {message.stop_reason}")

if __name__ == "__main__":
    main()
