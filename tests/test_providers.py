"""测试 Xunfei 和 Aliyun Provider 的异步 API 调用"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from openai import AsyncOpenAI


async def test_xunfei():
    """测试讯飞 API"""
    print("\n=== 测试讯飞 API ===")
    
    api_key = "51586d1fe075c5a29d5bf47c1a96a45f:YzMwZjcyZGZjYWI3ZjczYzFlYWI0ZTM3"
    base_url = "http://maas-api.cn-huabei-1.xf-yun.com/v2"
    model = "xopglm5"
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=10.0,  # 10秒超时
    )
    
    try:
        print(f"正在调用 {base_url} 模型 {model}...")
        print(f"API Key 前缀: {api_key[:20]}...")
        
        # 测试 health check
        print("\n1. 测试 health check (10秒超时)...")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
            temperature=0,
            max_tokens=10,
        )
        print(f"✅ Health check 成功")
        print(f"   Response: {response.choices[0].message.content}")
        
        # 测试实际分析
        print("\n2. 测试情节分析...")
        test_text = "叶凡醒来，发现自己身处荒凉之地。"
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output strict JSON."},
                {"role": "user", "content": f"Summarize plot in JSON with key: summary.\n\n{test_text}"},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=2048,
            extra_body={
                "response_format": {"type": "json_object"},
                "search_disable": True,
            },
        )
        print(f"✅ 情节分析成功")
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        
        return True
        
    except asyncio.TimeoutError as e:
        print(f"❌ 讯飞 API 超时 (10秒): {e}")
        print(f"   可能原因: 1) 网络连接问题 2) API 服务不可用 3) 防火墙阻止")
        return False
    except Exception as e:
        print(f"❌ 讯飞 API 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_aliyun():
    """测试阿里云 API"""
    print("\n=== 测试阿里云 API ===")
    
    api_key = "sk-7ee90114ef4e403f89b1f4b61ee776a3"
    # 阿里云通义千问 OpenAI 兼容接口正确路径
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "qwen-turbo"
    
    print(f"测试 URL 1: {base_url}")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=10.0,  # 10秒超时
    )
    
    try:
        print(f"正在调用 {base_url} 模型 {model}...")
        print(f"API Key 前缀: {api_key[:20]}...")
        
        # 测试 health check
        print("\n1. 测试 health check (10秒超时)...")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
            temperature=0,
            max_tokens=10,
        )
        print(f"✅ Health check 成功")
        print(f"   Response: {response.choices[0].message.content}")
        
        # 测试实际分析
        print("\n2. 测试情节分析...")
        test_text = "叶凡醒来，发现自己身处荒凉之地。"
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output strict JSON."},
                {"role": "user", "content": f"Summarize plot in JSON with key: summary.\n\n{test_text}"},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=2048,
        )
        print(f"✅ 情节分析成功")
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        
        return True
        
    except asyncio.TimeoutError as e:
        print(f"❌ 阿里云 API 超时 (10秒): {e}")
        print(f"   可能原因: 1) 网络连接问题 2) API 服务不可用")
        return False
    except Exception as e:
        print(f"❌ 阿里云 API 失败: {type(e).__name__}: {e}")
        
        # 尝试备用 URL
        print(f"\n尝试备用 URL...")
        alt_base_url = "https://dashscope.aliyuncs.com/compatible/openai/v1"
        print(f"测试 URL 2: {alt_base_url}")
        
        alt_client = AsyncOpenAI(
            api_key=api_key,
            base_url=alt_base_url,
            timeout=10.0,
        )
        
        try:
            response = await alt_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
                temperature=0,
                max_tokens=10,
            )
            print(f"✅ 备用 URL 成功！")
            print(f"   正确的 base_url 应该是: {alt_base_url}")
            return True
        except Exception as e2:
            print(f"❌ 备用 URL 也失败: {type(e2).__name__}: {e2}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """主测试函数"""
    print("开始测试 Provider API 调用...")
    print("=" * 60)
    
    xunfei_ok = await test_xunfei()
    aliyun_ok = await test_aliyun()
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  讯飞 API: {'✅ 通过' if xunfei_ok else '❌ 失败'}")
    print(f"  阿里云 API: {'✅ 通过' if aliyun_ok else '❌ 失败'}")
    
    if xunfei_ok and aliyun_ok:
        print("\n✅ 所有 Provider 测试通过！")
        return 0
    else:
        print("\n❌ 部分或全部 Provider 测试失败")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
