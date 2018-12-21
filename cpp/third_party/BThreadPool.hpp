#pragma once
#include <thread>
#include <memory>
#include <atomic>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>
static const std::size_t bthreadpoll_max_threads = 30;
class BThreadPool
{
public:
	using Task = std::function<void()>;
	/** 构造函数
	* @param numThreads[in] - 线程池中线程的个数
	*/
	explicit BThreadPool(int numThreads = 4) : mNumThreads(numThreads), mbStop(false)
	{
		if (mNumThreads > bthreadpoll_max_threads)
			mNumThreads = bthreadpoll_max_threads;
		mMaxQueueSize = mNumThreads * 10;
		StartAllThread(mNumThreads);
	}
	/**析构函数
	*/
	~BThreadPool()
	{
		StopAllThread();
	}
	/**添加任务
	*/
	template<class F, class... Args>
	auto AddTask(F&& f, Args&&... args)
		-> std::future<typename std::result_of<F(Args...)>::type>
	{
		//1. 返回类型
		using result_type = typename std::result_of<F(Args...)>::type;
		//2. 封装task
		auto task = std::make_shared< std::packaged_task<result_type()> >(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...));
		////3. 获取返回值
		auto res = task->get_future();
		{
			//4. 向队列中添加任务
			//4.1 先上锁
			std::unique_lock<std::mutex> lck(mTaskMutex);
			//4.2 查询是否可以添加
			mTaskCondFull.wait(lck, [this]()
			{
				return mbStop || mTaskQueue.size() < mMaxQueueSize;
			});
			//4.3 向队列中加入
			if (mbStop)
			{
				//在添加时，不允许退出线程池
				throw std::runtime_error("AddTask on stopped ThreadPool");
			}
			mTaskQueue.emplace([task]() {(*task)(); });
		}
		////5. 发送通知
		mTaskCondEmpty.notify_one();
		return res;
	}

	std::vector<std::thread::id> GetThreadIds()
	{
		std::vector<std::thread::id> ids;
		for (auto& i : mWorkers)
			ids.push_back(i.get_id());
		return ids;
	}
private:
	/**启动所有线程
	*/
	void StartAllThread(int numThreads)
	{
		for (int i = 0; i < numThreads; i++)
		{
			mWorkers.emplace_back(std::thread(&BThreadPool::Runing, this));
		}
		mbStop = false;
	}
	/**线程运行
	*/
	void Runing()
	{
		while (!mbStop)
		{
			Task task;
			//1. 从队列中取出任务
			{
				//1.1 上锁
				std::unique_lock<std::mutex> lck(mTaskMutex);
				//1.2 查询 
				mTaskCondEmpty.wait(lck, [this]() {
					return mbStop || (!mTaskQueue.empty());
				});
				//1.3 取出
				if (mbStop)
					break;
				task = std::move(mTaskQueue.front());
				mTaskQueue.pop();
				mTaskCondFull.notify_one();
			}
			//2. 运行这个任务
			task();
		}
	}
	/**停止所有线程
	*/
	void StopAllThread()
	{
		{
			std::unique_lock<std::mutex> lck(mTaskMutex);
			mbStop = true;
		}
		mTaskCondFull.notify_all();
		mTaskCondEmpty.notify_all();
		for (auto& w : mWorkers)
			w.join();
	}
private:
	int mNumThreads;                //线程池中线程的个数
	int mMaxQueueSize;                //线程池中，任务队列的最大值
	std::vector<std::thread>        mWorkers;        //工作线程
	std::queue<Task>                mTaskQueue;        //任务队列
	std::mutex                        mTaskMutex;        //同步锁
	std::condition_variable            mTaskCondEmpty;        //
	std::condition_variable            mTaskCondFull;        //
	std::atomic<bool>                mbStop;
};
