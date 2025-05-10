import { useState } from 'react';
import { Home, DollarSign, Check, AlertCircle, Info } from 'lucide-react';

export default function App() {
  const [features, setFeatures] = useState({
    area: '',
    frontage: '',
    accessRoad: '',
    houseDirection: '',
    balconyDirection: '',
    floors: '',
    bedrooms: '',
    bathrooms: '',
    legalStatus: '',
    furnitureState: ''
  });

  const [price, setPrice] = useState(null);
  const [priceInBillions, setPriceInBillions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Define the API URL - adjust this based on your environment
  const API_URL = 'http://localhost:5000';

  const handleChange = (field, value) => {
    setFeatures(prev => ({ ...prev, [field]: value }));
    setError(null);
    setPrice(null);
    setPriceInBillions(null);
  };

  // Validate inputs before proceeding
  const validateInputs = () => {
    if (!features.area || isNaN(parseFloat(features.area)) || parseFloat(features.area) <= 0) {
      setError('Vui lòng nhập diện tích hợp lệ');
      return false;
    }
    if (!features.frontage || isNaN(parseFloat(features.frontage)) || parseFloat(features.frontage) <= 0) {
      setError('Vui lòng nhập chiều rộng mặt tiền hợp lệ');
      return false;
    }
    if (!features.accessRoad || isNaN(parseFloat(features.accessRoad)) || parseFloat(features.accessRoad) <= 0) {
      setError('Vui lòng nhập chiều rộng đường vào hợp lệ');
      return false;
    }
    if (!features.houseDirection) {
      setError('Vui lòng chọn hướng nhà');
      return false;
    }
    if (!features.floors || isNaN(parseInt(features.floors)) || parseInt(features.floors) <= 0) {
      setError('Vui lòng nhập số tầng hợp lệ');
      return false;
    }
    if (!features.bedrooms || isNaN(parseInt(features.bedrooms)) || parseInt(features.bedrooms) < 0) {
      setError('Vui lòng nhập số phòng ngủ hợp lệ');
      return false;
    }
    if (!features.bathrooms || isNaN(parseInt(features.bathrooms)) || parseInt(features.bathrooms) < 0) {
      setError('Vui lòng nhập số phòng tắm hợp lệ');
      return false;
    }
    if (!features.legalStatus) {
      setError('Vui lòng chọn tình trạng pháp lý');
      return false;
    }
    if (!features.furnitureState) {
      setError('Vui lòng chọn tình trạng nội thất');
      return false;
    }

    return true;
  };

  // Handle prediction API call
  const handlePredict = async () => {
    if (!validateInputs()) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Prepare data for the API call
      const requestData = {
        area: features.area,
        frontage: features.frontage,
        accessRoad: features.accessRoad,
        houseDirection: features.houseDirection,
        balconyDirection: features.balconyDirection || features.houseDirection, // Set balcony direction to house direction if not provided
        floors: features.floors,
        bedrooms: features.bedrooms,
        bathrooms: features.bathrooms,
        legalStatus: features.legalStatus,
        furnitureState: features.furnitureState
      };

      // Call the API
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        // Try to get error details from response
        const errorData = await response.json();
        throw new Error(errorData.error || 'API request failed');
      }

      const data = await response.json();
      
      // Update UI with prediction results
      setPrice(data.prediction);
      setPriceInBillions(data.predictionInBillions);
      
    } catch (error) {
      console.error("Error:", error);
      setError(`Có lỗi xảy ra: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFeatures({
      area: '',
      frontage: '',
      accessRoad: '',
      houseDirection: '',
      balconyDirection: '',
      floors: '',
      bedrooms: '',
      bathrooms: '',
      legalStatus: '',
      furnitureState: ''
    });
    setPrice(null);
    setPriceInBillions(null);
    setError(null);
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('vi-VN').format(price);
  };

  const directionOptions = ["Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"];
  const legalStatusOptions = ["Sổ đỏ", "Sổ hồng", "Giấy tờ hợp lệ", "Đang chờ sổ", "Khác"];
  const furnitureStateOptions = ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất", "Cao cấp"];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          {/* Header */}
          <div className="bg-blue-600 p-6">
            <div className="flex items-center">
              <Home className="text-white mr-2" size={28} />
              <h1 className="text-2xl font-bold text-white">Định giá bất động sản</h1>
            </div>
            <p className="text-blue-100 mt-1">Nhập thông tin để nhận định giá chính xác</p>
          </div>

          {/* Form */}
          <div className="p-6">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-4">Thông tin cơ bản</h2>
              
              {/* Basic measurements */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Diện tích (m²) <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="VD: 100"
                    value={features.area}
                    onChange={(e) => handleChange('area', e.target.value)}
                    min="1"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Mặt tiền (m) <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="VD: 5"
                    value={features.frontage}
                    onChange={(e) => handleChange('frontage', e.target.value)}
                    min="0.1"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Đường vào (m) <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="VD: 8"
                    value={features.accessRoad}
                    onChange={(e) => handleChange('accessRoad', e.target.value)}
                    min="0.1"
                    step="0.1"
                  />
                </div>
              </div>
            </div>

            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-4">Chi tiết thiết kế</h2>

              {/* Direction selection */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Hướng nhà <span className="text-red-500">*</span>
                  </label>
                  <select
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={features.houseDirection}
                    onChange={(e) => handleChange('houseDirection', e.target.value)}
                  >
                    <option value="">-- Chọn hướng --</option>
                    {directionOptions.map(direction => (
                      <option key={direction} value={direction}>{direction}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Hướng ban công
                  </label>
                  <select
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={features.balconyDirection}
                    onChange={(e) => handleChange('balconyDirection', e.target.value)}
                  >
                    <option value="">-- Không có ban công --</option>
                    {directionOptions.map(direction => (
                      <option key={direction} value={direction}>{direction}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Structure details */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Số tầng <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="VD: 3"
                    value={features.floors}
                    onChange={(e) => handleChange('floors', e.target.value)}
                    min="1"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Số phòng ngủ <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="VD: 2"
                    value={features.bedrooms}
                    onChange={(e) => handleChange('bedrooms', e.target.value)}
                    min="0"
                  />
                </div>
                <div>
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Số phòng tắm <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="VD: 2"
                    value={features.bathrooms}
                    onChange={(e) => handleChange('bathrooms', e.target.value)}
                    min="0"
                  />
                </div>
              </div>
            </div>

            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-4">Pháp lý & Nội thất</h2>

              {/* Legal status */}
              <div className="mb-4">
                <label className="block text-gray-700 text-sm font-medium mb-2">
                  Tình trạng pháp lý <span className="text-red-500">*</span>
                </label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {legalStatusOptions.map(status => (
                    <div
                      key={status}
                      className={`cursor-pointer border rounded-lg p-3 transition-colors ${
                        features.legalStatus === status
                          ? 'bg-blue-50 border-blue-500 text-blue-700'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                      onClick={() => handleChange('legalStatus', status)}
                    >
                      <div className="flex items-center">
                        <div className={`w-4 h-4 rounded-full mr-2 border ${
                          features.legalStatus === status
                            ? 'bg-blue-500 border-blue-500'
                            : 'border-gray-400'
                        }`}>
                          {features.legalStatus === status && (
                            <Check size={14} className="text-white" />
                          )}
                        </div>
                        <span>{status}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Furniture state */}
              <div className="mb-4">
                <label className="block text-gray-700 text-sm font-medium mb-2">
                  Tình trạng nội thất <span className="text-red-500">*</span>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {furnitureStateOptions.map(state => (
                    <div
                      key={state}
                      className={`cursor-pointer border rounded-lg p-3 transition-colors ${
                        features.furnitureState === state
                          ? 'bg-blue-50 border-blue-500 text-blue-700'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                      onClick={() => handleChange('furnitureState', state)}
                    >
                      <div className="flex items-center">
                        <div className={`w-4 h-4 rounded-full mr-2 border ${
                          features.furnitureState === state
                            ? 'bg-blue-500 border-blue-500'
                            : 'border-gray-400'
                        }`}>
                          {features.furnitureState === state && (
                            <Check size={14} className="text-white" />
                          )}
                        </div>
                        <span>{state}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Error message */}
            {error && (
              <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg flex items-start">
                <AlertCircle className="mr-2 flex-shrink-0 mt-0.5" size={16} />
                <span>{error}</span>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex justify-between mt-6">
              <button
                className="bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-2 px-6 rounded-lg transition-colors"
                onClick={handleReset}
              >
                Nhập lại
              </button>
              <button
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors flex items-center"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? (
                  <>Đang tính toán...</>
                ) : (
                  <>
                    <DollarSign className="mr-2" size={18} />
                    Định giá bất động sản
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Section */}
          {price !== null && (
            <div className="border-t border-gray-200 bg-gray-50 p-6">
              <h2 className="text-xl font-semibold mb-4">Kết quả định giá</h2>

              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Giá ước tính</p>
                    <div className="text-3xl font-bold text-blue-600">{formatPrice(price)} VNĐ</div>
                    <p className="text-sm text-gray-500 mt-1">({priceInBillions?.toFixed(2)} tỷ đồng)</p>
                  </div>
                </div>

                
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}