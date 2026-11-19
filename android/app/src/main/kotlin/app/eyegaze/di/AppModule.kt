package app.eyegaze.di

import app.eyegaze.runtime.DefaultGazeRuntimeFactory
import app.eyegaze.runtime.GazeRuntimeFactory
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import javax.inject.Qualifier

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class DefaultDispatcher

@Module
@InstallIn(SingletonComponent::class)
abstract class AppModule {
    @Binds
    abstract fun gazeRuntimeFactory(impl: DefaultGazeRuntimeFactory): GazeRuntimeFactory

    companion object {
        @Provides
        @DefaultDispatcher
        fun defaultDispatcher(): CoroutineDispatcher = Dispatchers.Default
    }
}
