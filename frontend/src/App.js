import React, { useState } from "react";
import "./index.css";

function App() {
  const [file, setFile] = useState(null);
  const [categories, setCategories] = useState({
    apple: false,
    banana: false,
    orange: false,
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [resultUrl, setResultUrl] = useState(null);
  const [error, setError] = useState(null);
  const [fruitCounts, setFruitCounts] = useState({});

  const isButtonDisabled = !file || !Object.values(categories).some((val) => val);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResultUrl(null);
    setError(null);
    setFruitCounts({});
    console.log("Selected file:", selectedFile?.name, "Type:", selectedFile?.type);
  };

  const handleCategoryChange = (e) => {
    setCategories({ ...categories, [e.target.name]: e.target.checked });
  };

  const handleProcess = async () => {
    if (!file || isButtonDisabled) return;
    setIsProcessing(true);
    setError(null);
    setResultUrl(null);
    setFruitCounts({});

    const formData = new FormData();
    formData.append("file", file);
    formData.append("categories", JSON.stringify(Object.keys(categories).filter((key) => categories[key])));

    try {
      const response = await fetch("http://localhost:8000/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ошибка сервера: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      console.log("Backend response:", result);
      setResultUrl(`http://localhost:8000${result.output_url}`);
      setFruitCounts(result.fruit_counts);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message || "Не удалось обработать файл. Проверьте сервер.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-gray-100 min-h-screen flex items-center justify-center">
      <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
        <h1 className="text-2xl font-bold mb-6 text-center">Fruit Detection</h1>

        {/* Загрузка файла */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload Image or Video
          </label>
          <input
            type="file"
            accept="image/*,video/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>

        {/* Чекбоксы для категорий */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Fruit Categories
          </label>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                name="apple"
                checked={categories.apple}
                onChange={handleCategoryChange}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-700">Apples</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                name="banana"
                checked={categories.banana}
                onChange={handleCategoryChange}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-700">Bananas</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                name="orange"
                checked={categories.orange}
                onChange={handleCategoryChange}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-700">Oranges</span>
            </label>
          </div>
        </div>

        {/* Кнопка обработки */}
        <button
          onClick={handleProcess}
          disabled={isButtonDisabled || isProcessing}
          className={`w-full py-2 px-4 rounded-md text-white font-semibold ${
            isButtonDisabled || isProcessing ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {isProcessing ? "Processing..." : "Process"}
        </button>

        {/* Отображение ошибки */}
        {error && (
          <div className="mt-4 text-red-600 text-sm text-center">
            {error}
          </div>
        )}

        {/* Отображение результата */}
        {resultUrl && (
          <div className="mt-6">
            <h2 className="text-lg font-medium text-gray-700 mb-2">Result</h2>
            {file?.type.startsWith("image") ? (
              <img
                src={resultUrl}
                alt="Processed Image"
                className="w-full rounded-md"
                onError={(e) => console.error("Image load error:", e)}
              />
            ) : (
              <video
                controls
                src={resultUrl}
                className="w-full rounded-md"
                onError={(e) => console.error("Video load error:", e)}
                type="video/mp4"
              />
            )}
            {Object.keys(fruitCounts).length > 0 && (
              <div className="mt-4">
                <h3 className="text-md font-medium text-gray-700">Fruit Counts:</h3>
                <ul className="list-disc list-inside">
                  {Object.entries(fruitCounts).map(([fruit, count]) => (
                    <li key={fruit} className="text-sm text-gray-600">
                      {fruit}: {count}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;